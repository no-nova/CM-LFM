// Phase 3 CPU kernels for LFM3Conv (Zen 5 / AVX-512, bf16 + IQ4_NL).
//
// All modes run on the *same* real layer-0 tensors and real activations dumped by
// scripts/phase3_dump.py, so the only difference is the kernel/data layout.
//
//   unfused_bf16          the PyTorch op sequence with every intermediate
//                         materialised (in_proj -> transpose copy -> two gate
//                         passes -> conv1d -> transpose+contiguous -> out_proj)
//   fused_bf16            one pass: in_proj + pre-gate -> causal depthwise conv +
//                         post-gate -> 32-lane block energy -> out_proj, with no
//                         intermediate round trips
//   fused_out_iq4_dense   same fused pass, out_proj streamed as 4.5-bit IQ4_NL
//                         (18 B / 32 weights) in 64-byte-aligned block-major layout
//   fused_out_iq4_sparse  same, but block slabs whose L2 energy is below the
//                         calibrated theta are never read (Variant A: state-exact)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>
#include <immintrin.h>

static const int DB = 32;    // lanes per 4.5-bit block
static const int QB = 18;    // bytes per block: 16 nibble bytes + fp16 scale
static const float KV[16] = {-127.f, -104.f, -83.f, -65.f, -45.f, -28.f, -12.f, 1.f,
                             13.f, 23.f, 33.f, 41.f, 49.f, 56.f, 62.f, 66.f};

static inline float bf2f(uint16_t v) { uint32_t u = uint32_t(v) << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t(u >> 16); }

static std::vector<uint16_t> load_bf16(const std::string& p, size_t n_expect) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (n_expect && sz != n_expect * 2) {
        fprintf(stderr, "%s: %zu bytes, expected %zu\n", p.c_str(), sz, n_expect * 2); exit(1);
    }
    std::vector<uint16_t> v(sz / 2);
    if (fread(v.data(), 2, v.size(), f) != v.size()) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

static inline float dot_bf16(const uint16_t* w, const uint16_t* x, int n) {
    __m512 acc = _mm512_setzero_ps();
    int j = 0;
    for (; j + 32 <= n; j += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((const void*)(w + j));
        __m512bh xv = (__m512bh)_mm512_loadu_si512((const void*)(x + j));
        acc = _mm512_dpbf16_ps(acc, wv, xv);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) s += bf2f(w[j]) * bf2f(x[j]);
    return s;
}

// ------------------------------------------------------------- IQ4_NL pack
static void pack_iq4_blockmajor(const std::vector<uint16_t>& w, int n_out, int n_in,
                                std::vector<uint8_t>& out) {
    int nb = n_in / DB;
    out.assign(size_t(nb) * n_out * QB, 0);
    for (int b = 0; b < nb; ++b) {
        for (int o = 0; o < n_out; ++o) {
            const uint16_t* row = w.data() + size_t(o) * n_in + b * DB;
            float amax = 0.f;
            for (int i = 0; i < DB; ++i) amax = std::max(amax, std::fabs(bf2f(row[i])));
            float d = amax / 127.f, inv = d > 0 ? 1.f / d : 0.f;
            uint8_t codes[DB];
            for (int i = 0; i < DB; ++i) {
                float t = bf2f(row[i]) * inv;
                int best = 0; float bd = 1e30f;
                for (int c = 0; c < 16; ++c) { float dd = std::fabs(t - KV[c]); if (dd < bd) { bd = dd; best = c; } }
                codes[i] = uint8_t(best);
            }
            uint8_t* p = out.data() + (size_t(b) * n_out + o) * QB;
            for (int i = 0; i < 16; ++i) p[i] = uint8_t(codes[i] | (codes[i + 16] << 4));
            uint16_t dh = f2bf(d);
            memcpy(p + 16, &dh, 2);
        }
    }
}

static inline __m128i iq4_table() {
    signed char t[16];
    for (int i = 0; i < 16; ++i) t[i] = (signed char)KV[i];
    return _mm_loadu_si128((const __m128i*)t);
}

// Decode 32 4.5-bit codes to two f32x16 vectors with pshufb (1 op / 16 codes).
static inline void iq4_decode32(const uint8_t* p, const __m128i tbl, __m512& w0, __m512& w1) {
    __m128i nb = _mm_loadu_si128((const __m128i*)p);
    __m128i lo = _mm_and_si128(nb, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(nb, 4), _mm_set1_epi8(0x0F));
    __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(tbl, lo)));
    __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(tbl, hi)));
    uint16_t dh; memcpy(&dh, p + 16, 2);
    __m512 d = _mm512_set1_ps(bf2f(dh));
    w0 = _mm512_mul_ps(f0, d);
    w1 = _mm512_mul_ps(f1, d);
}

// One 32-lane block dot against a float row, accumulated into a 32-lane vector
// so the horizontal reduction happens once per output row, not once per block.
static inline void dot_iq4_acc(const uint8_t* p, const float* x, const __m128i tbl, __m512& acc) {
    __m512 w0, w1;
    iq4_decode32(p, tbl, w0, w1);
    acc = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x), acc);
    acc = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x + 16), acc);
}

static inline float dot_iq4(const uint8_t* p, const float* x) {
    __m512 acc = _mm512_setzero_ps();
    dot_iq4_acc(p, x, iq4_table(), acc);
    return _mm512_reduce_add_ps(acc);
}

static int active_blocks(const float* y, int d, float theta, int* act) {
    int nb = d / DB, n = 0;
    for (int b = 0; b < nb; ++b) {
        float e = 0.f;
        for (int i = 0; i < DB; ++i) { float t = y[b * DB + i]; e += t * t; }
        if (!(e < theta)) act[n++] = b;
    }
    return n;
}

struct Layer {
    int d = 2048, k = 7, T = 64, n_in = 2048;
    std::vector<uint16_t> w_in, cw, cb, w_out, x;
    std::vector<uint8_t> q_out;
    // active pointers (allow a DRAM-cold pool of weight replicas)
    const uint16_t *W_in = nullptr, *W_out = nullptr;
    const uint8_t *Q_out = nullptr;
    std::vector<float> h2, cbuf;         // [channel][token] for the depthwise conv
    std::vector<float> yb;               // [token][channel] gated output, row contiguous
    std::vector<uint16_t> ybf;           // same, bf16, for the bf16 out_proj path
    float theta = 0.f;
};

enum Kind { FUSED_BF16 = 0, FUSED_IQ4 = 1, FUSED_IQ4_SPARSE = 2 };

static void fused(Layer& L, Kind kind, int threads, std::vector<float>& out) {
    const int d = L.d, T = L.T, k = L.k;
    L.h2.assign(size_t(d) * T, 0.f);
    L.cbuf.assign(size_t(d) * T, 0.f);
    L.yb.assign(size_t(T) * d, 0.f);
    L.ybf.assign(size_t(T) * d, 0);
    out.assign(size_t(T) * d, 0.f);

    // stage 1: three input projections, pre-gate, no transposed re-read
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int t = 0; t < T; ++t) {
        const uint16_t* xr = L.x.data() + size_t(t) * L.n_in;
        for (int c = 0; c < d; ++c) {
            float B = dot_bf16(L.W_in + size_t(c) * L.n_in, xr, L.n_in);
            float X = dot_bf16(L.W_in + size_t(2 * d + c) * L.n_in, xr, L.n_in);
            L.h2[size_t(c) * T + t] = B * X;
            L.cbuf[size_t(c) * T + t] = dot_bf16(L.W_in + size_t(d + c) * L.n_in, xr, L.n_in);
        }
    }
    // stage 2: causal depthwise short conv + post-gate
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int c = 0; c < d; ++c) {
        const uint16_t* wk = L.cw.data() + size_t(c) * k;
        const float* hr = L.h2.data() + size_t(c) * T;
        for (int t = 0; t < T; ++t) {
            float s = bf2f(L.cb[c]);
            for (int j = 0; j < k; ++j) { int u = t - j; if (u >= 0) s += bf2f(wk[j]) * hr[u]; }
            float v = s * L.cbuf[size_t(c) * T + t];
            L.yb[size_t(t) * d + c] = v;
            L.ybf[size_t(t) * d + c] = f2bf(v);
        }
    }
    // stage 3: out_proj, optionally IQ4_NL block-major, optionally block-skipped
    const int nb = d / DB;
    std::vector<unsigned char> keepb(size_t(T) * nb, 1);
    if (kind == FUSED_IQ4_SPARSE) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < T; ++t) {
            const float* yr = L.yb.data() + size_t(t) * d;
            unsigned char* kp = keepb.data() + size_t(t) * nb;
            for (int b = 0; b < nb; ++b) {
                float e = 0.f;
                for (int i = 0; i < DB; ++i) { float v = yr[b * DB + i]; e += v * v; }
                kp[b] = (e < L.theta) ? 0 : 1;
            }
        }
    }
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int t = 0; t < T; ++t) {
        float* orow = out.data() + size_t(t) * d;
        if (kind == FUSED_BF16) {
            const uint16_t* yr = L.ybf.data() + size_t(t) * d;
            for (int o = 0; o < d; ++o)
                orow[o] = dot_bf16(L.W_out + size_t(o) * d, yr, d);
        } else {
            const unsigned char* kp = keepb.data() + size_t(t) * nb;
            const bool sp = (kind == FUSED_IQ4_SPARSE);
            const __m128i tbl = iq4_table();
            const float* yr = L.yb.data() + size_t(t) * d;
            const int OG = 8;
            for (int o0 = 0; o0 < d; o0 += OG) {
                __m512 acc[OG];
                for (int i = 0; i < OG; ++i) acc[i] = _mm512_setzero_ps();
                for (int b = 0; b < nb; ++b) {
                    if (sp && !kp[b]) continue;
                    const float* xb = yr + b * DB;
                    const uint8_t* slab = L.Q_out + size_t(b) * d * QB;
                    for (int u = 0; u < OG; ++u)
                        dot_iq4_acc(slab + size_t(o0 + u) * QB, xb, tbl, acc[u]);
                }
                for (int i = 0; i < OG; ++i) orow[o0 + i] = _mm512_reduce_add_ps(acc[i]);
            }
        }
    }
}

static inline float dot_bf16f(const uint16_t* w, const float* x, int n) {
    __m512 acc = _mm512_setzero_ps();
    int j = 0;
    alignas(64) uint16_t xb[32];
    for (; j + 32 <= n; j += 32) {
        for (int i = 0; i < 32; ++i) xb[i] = f2bf(x[j + i]);
        __m512bh wv = (__m512bh)_mm512_loadu_si512((const void*)(w + j));
        acc = _mm512_dpbf16_ps(acc, wv, (__m512bh)_mm512_loadu_si512((const void*)xb));
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; j < n; ++j) s += bf2f(w[j]) * x[j];
    return s;
}

// The PyTorch op sequence with every intermediate materialised.
static void unfused_ref(Layer& L, int threads, std::vector<float>& out) {
    const int d = L.d, T = L.T, k = L.k;
    std::vector<float> bcx(size_t(3 * d) * T), h(size_t(d) * T), y(size_t(d) * T), yt(size_t(T) * d);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int t = 0; t < T; ++t) {
        const uint16_t* xr = L.x.data() + size_t(t) * L.n_in;
        for (int o = 0; o < 3 * d; ++o)
            bcx[size_t(o) * T + t] = dot_bf16(L.W_in + size_t(o) * L.n_in, xr, L.n_in);
    }
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < d; ++c)
            h[size_t(c) * T + t] = bcx[size_t(c) * T + t] * bcx[size_t(2 * d + c) * T + t];
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int c = 0; c < d; ++c) {
        const uint16_t* wk = L.cw.data() + size_t(c) * k;
        for (int t = 0; t < T; ++t) {
            float s = bf2f(L.cb[c]);
            for (int j = 0; j < k; ++j) { int u = t - j; if (u >= 0) s += bf2f(wk[j]) * h[size_t(c) * T + u]; }
            y[size_t(c) * T + t] = s;
        }
    }
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < d; ++c)
            yt[size_t(t) * d + c] = bcx[size_t(d + c) * T + t] * y[size_t(c) * T + t];
    out.assign(size_t(T) * d, 0.f);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < d; ++o)
            out[size_t(t) * d + o] = dot_bf16f(L.W_out + size_t(o) * d, yt.data() + size_t(t) * d, d);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "artifacts/p3";
    int threads = argc > 2 ? atoi(argv[2]) : 1;
    int reps = argc > 3 ? atoi(argv[3]) : 15;
    Layer L;
    L.w_in = load_bf16(dir + "/p3_in.bin", size_t(3 * L.d) * L.n_in);
    L.cw = load_bf16(dir + "/p3_cw.bin", size_t(L.d) * L.k);
    L.cb = load_bf16(dir + "/p3_cb.bin", size_t(L.d));
    L.w_out = load_bf16(dir + "/p3_out.bin", size_t(L.d) * L.d);
    L.x = load_bf16(dir + "/p3_x.bin", size_t(64) * L.n_in);
    const int Tsel = argc > 5 ? atoi(argv[5]) : L.T;
    const int xoff = argc > 6 ? atoi(argv[6]) : 0;
    L.T = Tsel;
    if (xoff > 0) {
        std::vector<uint16_t> xw(L.x.begin() + size_t(xoff) * L.n_in,
                                 L.x.begin() + size_t(xoff + Tsel) * L.n_in);
        L.x = xw;
    } else {
        L.x.resize(size_t(Tsel) * L.n_in);
    }
    pack_iq4_blockmajor(L.w_out, L.d, L.d, L.q_out);
    L.W_in = L.w_in.data(); L.W_out = L.w_out.data(); L.Q_out = L.q_out.data();
    const int nb = L.d / DB;
    const int POOL = argc > 4 ? atoi(argv[4]) : 1;   // >1 => DRAM-cold replicas
    std::vector<std::vector<uint16_t>> pin, pout;
    std::vector<std::vector<uint8_t>> pq;
    for (int k = 1; k < POOL; ++k) {
        pin.push_back(L.w_in); pout.push_back(L.w_out); pq.push_back(L.q_out);
    }
    const double b_in = double(3 * L.d) * L.n_in * 2, b_out = double(L.d) * L.d * 2;
    const double b_q = double(L.d / DB) * L.d * QB;

    std::vector<float> a, b;
    L.theta = 0.f;
    unfused_ref(L, threads, a);
    fused(L, FUSED_BF16, threads, b);
    double maxd = 0;
    for (size_t i = 0; i < a.size(); ++i) maxd = std::max(maxd, (double)std::fabs(a[i] - b[i]));

    auto bench = [&](const char* name, Kind kind, bool un) {
        std::vector<double> v;
        for (int r = 0; r < reps + POOL - 1; ++r) {
            if (POOL > 1) {
                int k = r % POOL;
                L.W_in = k ? pin[k - 1].data() : L.w_in.data();
                L.W_out = k ? pout[k - 1].data() : L.w_out.data();
                L.Q_out = k ? pq[k - 1].data() : L.q_out.data();
            }
            if (r < POOL - 1) continue;
            double t0 = now_ms();
            std::vector<float> o;
            if (un) unfused_ref(L, threads, o); else fused(L, kind, threads, o);
            v.push_back(now_ms() - t0);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    // theta: either the Phase-2 calibrated value (calib=0), or a per-shape
    // calibration at the requested tau (calib>0) -- decode has a warm conv state
    // in deployment while this harness replays from a cold one, so the shapes do
    // not share a threshold.
    const double calib = argc > 7 ? atof(argv[7]) : 0.0;
    L.theta = 2.6106834411621094e-05f;
    if (calib > 0) {
        std::vector<float> es;
        { // energies of the gated output at this exact shape
            std::vector<float> dummy; L.theta = 0.f;
            fused(L, FUSED_BF16, threads, dummy);
            for (int t = 0; t < L.T; ++t)
                for (int b = 0; b < nb; ++b) {
                    float e = 0;
                    for (int u = 0; u < DB; ++u) { float v = L.yb[size_t(t) * L.d + b * DB + u]; e += v * v; }
                    es.push_back(e);
                }
        }
        std::sort(es.begin(), es.end());
        double q = 1.0 - calib;
        size_t idx = std::min(es.size() - 1, size_t(q * (es.size() - 1)));
        L.theta = es[idx];
    }
    double ms_un = bench("unfused_bf16", FUSED_BF16, true);
    double ms_fu = bench("fused_bf16", FUSED_BF16, false);
    double ms_iq = bench("fused_iq4", FUSED_IQ4, false);
    double ms_sp = bench("fused_iq4_sparse", FUSED_IQ4_SPARSE, false);

    // realised skip statistics on the real activations
    std::vector<int> act(nb);
    fused(L, FUSED_IQ4_SPARSE, threads, b);
    long keep = 0;
    for (int t = 0; t < L.T; ++t)
        keep += active_blocks(L.yb.data() + size_t(t) * L.d, L.d, L.theta, act.data());
    double frac = double(keep) / double(L.T * nb);

    double emean = 0, emin = 1e30, emax = 0;
    for (int t = 0; t < L.T; ++t)
        for (int b = 0; b < nb; ++b) {
            double e = 0;
            for (int u = 0; u < DB; ++u) { double v = L.yb[size_t(t) * L.d + b * DB + u]; e += v * v; }
            emean += e; emin = std::min(emin, e); emax = std::max(emax, e);
        }
    emean /= double(L.T * nb);
    printf("{\"threads\": %d, \"d\": %d, \"T\": %d, \"k\": %d, \"block_energy\": {\"mean\": %.6g, \"min\": %.6g, \"max\": %.6g},\n",
           threads, L.d, L.T, L.k, emean, emin, emax);
    printf(" \"bytes\": {\"in_proj_bf16\": %.0f, \"out_proj_bf16\": %.0f, \"out_proj_iq4\": %.0f},"
           " \"fused_vs_unfused_maxabs\": %.3g,\n", b_in, b_out, b_q, maxd);
    printf(" \"ms\": {\"unfused_bf16\": %.4f, \"fused_bf16\": %.4f, \"fused_iq4\": %.4f, \"fused_iq4_sparse\": %.4f},\n",
           ms_un, ms_fu, ms_iq, ms_sp);
    printf(" \"gbps_weights_streamed\": {\"unfused_bf16\": %.2f, \"fused_bf16\": %.2f, \"fused_iq4\": %.2f, \"fused_iq4_sparse\": %.2f},\n",
           L.T * (b_in + b_out) / (ms_un * 1e6), L.T * (b_in + b_out) / (ms_fu * 1e6),
           L.T * (b_in + b_q) / (ms_iq * 1e6), L.T * (b_in + frac * b_q) / (ms_sp * 1e6));
    printf(" \"skipped\": {\"theta\": %.6g, \"frac_blocks_active\": %.4f, \"out_proj_bytes_read\": %.0f},\n",
           double(L.theta), frac, frac * b_q);
    printf(" \"speedup\": {\"fused_vs_unfused\": %.3f, \"iq4_dense_vs_fused_bf16\": %.3f, \"iq4_sparse_vs_fused_bf16\": %.3f}}\n",
           ms_un / ms_fu, ms_fu / ms_iq, ms_fu / ms_sp);
    return 0;
}
