// Phase 3 (R3): decode-shaped GEMV for the non-gated projections.
//
//   ./p3gqa <dir> <reps> [threads]
//
// Matrices (bf16, row-major native order, dumped by scripts/phase3_dump2.py):
//   q_proj 2048x2048, k_proj/v_proj 512x2048, o_proj 2048x2048,
//   lm_head 128000x2048 (tied to embed_tokens: 500 MiB, the largest single
//   tensor of the model and the biggest Amdahl term of CPU decode).
//
// Two modes per matrix:
//   bf16  : y[o] = sum_i W[o][i] x[i]          (weights streamed once, bf16 dot)
//   iq4   : 4.5-bit IQ4_NL, *block-major* layout [block][out][18 B], so the slab
//           of one 32-lane block is n_out*18 contiguous bytes (lm_head: 2.30 MiB,
//           exactly 36 864 cache lines) and the 3.56x byte reduction is real.
//
// Cold measurement: the whole working set is evicted with a 96 MiB sweep
// between repetitions, so the numbers are DRAM, not L3.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <immintrin.h>
#include <omp.h>

static const int DB = 32;

static std::vector<uint16_t> load_bf16(const std::string &p, size_t n) {
    std::vector<uint16_t> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    if (fread(v.data(), 2, n, f) != n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}

static inline float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline __m512 bf16x16(const uint16_t *p) {          // bf16 -> f32 (NOT fp16!)
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    __m512i i = _mm512_cvtepu16_epi32(h);
    return _mm512_castsi512_ps(_mm512_slli_epi32(i, 16));
}

static float dot_bf16(const uint16_t *w, const float *x, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(bf16x16(w + i), _mm512_loadu_ps(x + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) s += bf2f(w[i]) * x[i];
    return s;
}

// ---------------------------------------------------------------- IQ4_NL 4.5 bit
// Symmetric 16-level codebook (16 B codes per 32 weights + one bf16 scale).
// The llama.cpp kvalues_iq4nl table was measured to cost 3.7x more reconstruction
// error on these weights (results/phase3_kernels2_nl.csv), see the report.
static const int8_t LUT[16] = {-127, -110, -93, -76, -59, -42, -25, -8,
                               8, 25, 42, 59, 76, 93, 110, 127};

static void pack_iq4(const std::vector<uint16_t> &W, int n_out, int d,
                     std::vector<uint8_t> &Q) {
    const int nb = d / DB;
    Q.assign((size_t)nb * n_out * 18, 0);
    for (int b = 0; b < nb; ++b) {
        for (int o = 0; o < n_out; ++o) {
            const uint16_t *w = &W[(size_t)o * d + b * DB];
            float v[DB], amax = 0.f;
            for (int i = 0; i < DB; ++i) {
                v[i] = bf2f(w[i]);
                amax = std::max(amax, std::fabs(v[i]));
            }
            float scale = amax > 0.f ? amax / 127.f : 1.f;
            float inv = 1.f / scale;
            uint8_t *slot = &Q[((size_t)b * n_out + o) * 18];
            int best[DB];
            for (int i = 0; i < DB; ++i) {
                float q = v[i] * inv;
                int bi = 0; float bd = 1e30f;
                for (int k = 0; k < 16; ++k) {
                    float dd = std::fabs(q - LUT[k]);
                    if (dd < bd) { bd = dd; bi = k; }
                }
                best[i] = bi;
            }
            for (int k = 0; k < 16; ++k)   // 16 bytes carry the 32 codes
                slot[k] = (uint8_t)((best[2 * k] & 15) | ((best[2 * k + 1] & 15) << 4));
            uint32_t bits;
            memcpy(&bits, &scale, 4);
            uint16_t sb = (uint16_t)(bits >> 16);   // bf16 = top half of the f32
            memcpy(slot + 16, &sb, 2);
        }
    }
}

// the table lives in the *float* domain: codebook entries are real weights
static inline __m512 lut16() {
    return _mm512_setr_ps(LUT[0], LUT[1], LUT[2], LUT[3], LUT[4], LUT[5], LUT[6], LUT[7],
                          LUT[8], LUT[9], LUT[10], LUT[11], LUT[12], LUT[13], LUT[14], LUT[15]);
}

// 16 nibbles -> 16 f32: one vpermps against the 16-lane float table, then scale
static inline __m512 decode16(__m128i codes, __m512 scale, __m512 lut) {
    __m512i idx = _mm512_cvtepu8_epi32(codes);
    return _mm512_mul_ps(_mm512_permutexvar_ps(idx, lut), scale);
}

// one 18-byte block slot (32 codes + bf16 scale) dot one 32-lane x
static inline float dot_iq4(const uint8_t *slot, const float *x) {
    uint16_t sb;
    memcpy(&sb, slot + 16, 2);
    __m512 scale = _mm512_set1_ps(bf2f(sb));
    __m512 lut = lut16();
    __m128i c = _mm_loadu_si128((const __m128i *)slot);
    const __m128i m = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(c, m), hi = _mm_and_si128(_mm_srli_epi16(c, 4), m);
    __m128i c0 = _mm_unpacklo_epi8(lo, hi);          // codes 0..15
    __m128i c1 = _mm_unpackhi_epi8(lo, hi);          // codes 16..31
    __m512 acc = _mm512_mul_ps(decode16(c0, scale, lut), _mm512_loadu_ps(x));
    acc = _mm512_fmadd_ps(decode16(c1, scale, lut), _mm512_loadu_ps(x + 16), acc);
    return _mm512_reduce_add_ps(acc);
}

static std::vector<float> bench_flush;   // 96 MiB eviction buffer

static void flush_l3() {
    volatile float s = 0.f;
    for (size_t i = 0; i < bench_flush.size(); i += 16) s += bench_flush[i];
    (void)s;
}

int main(int argc, char **argv) {
    std::string dir = argc > 1 ? argv[1] : "artifacts/p3";
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    int threads = argc > 3 ? atoi(argv[3]) : 16;
    omp_set_num_threads(threads);
    bench_flush.assign((96u << 20) / 4, 1.0f);

    struct Spec { const char *tag; int n_out, d; };
    Spec specs[] = {{"q", 2048, 2048}, {"k", 512, 2048}, {"v", 512, 2048},
                    {"o", 2048, 2048}, {"lm", 128000, 2048}};

    printf("threads,%d\n", threads);
    printf("matrix,n_out,d,mode,ms,gbps,bytes_MB\n");
    for (auto &sp : specs) {
        auto W = load_bf16(dir + "/p3_" + sp.tag + ".bin", (size_t)sp.n_out * sp.d);
        // Small matrices are L3-resident (64 MiB) and would measure cache speed, so
        // replicate them until the working set is clearly DRAM-bound.
        int pool = 1;
        {
            const double mb = (double)sp.n_out * sp.d * 2 / 1048576.0;
            while (pool < 64 && mb * pool < 200.0) ++pool;
            if (pool > 1) {
                std::vector<uint16_t> big(W);
                big.reserve(W.size() * pool);
                for (int k = 1; k < pool; ++k) big.insert(big.end(), W.begin(), W.end());
                W.swap(big);
            }
        }
        const int rows = sp.n_out * pool;
        std::vector<float> x(sp.d, 0.f);
        for (int i = 0; i < sp.d; ++i) x[i] = std::sin(0.01f * i) * 0.5f;
        std::vector<float> y(sp.n_out);
        const double bytes_bf16 = (double)sp.n_out * sp.d * 2.0;
        std::vector<double> tb, tq;
        double bytes_iq4 = (double)(sp.d / DB) * sp.n_out * 18.0;

        for (int r = 0; r < reps; ++r) {                       // bf16 dense
            flush_l3();
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
            for (int o = 0; o < rows; ++o)
                y[o % sp.n_out] = dot_bf16(&W[(size_t)o * sp.d], x.data(), sp.d);
            auto t1 = std::chrono::steady_clock::now();
            tb.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::vector<uint8_t> Q;
        pack_iq4(W, rows, sp.d, Q);
        std::vector<float> yq(sp.n_out, 0.f);
        const int nb = sp.d / DB;
        const int nt = omp_get_max_threads();
        std::vector<std::vector<float>> yt(nt, std::vector<float>(sp.n_out * pool, 0.f));
        for (int r = 0; r < reps; ++r) {                       // IQ4 block-major
            for (auto &v : yt) std::fill(v.begin(), v.end(), 0.f);
            flush_l3();
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                float *acc = yt[tid].data();
                __m512 lut = lut16();
                (void)lut;
#pragma omp for schedule(static)
                for (int b = 0; b < nb; ++b) {
                    const uint8_t *slab = &Q[(size_t)b * rows * 18];
                    const float *xb = x.data() + b * DB;
                    for (int o = 0; o < rows; ++o)
                        acc[o] += dot_iq4(slab + (size_t)o * 18, xb);
                }
            }
            for (auto &v : yt)
                for (int o = 0; o < rows; ++o) yq[o % sp.n_out] += v[o];
            auto t1 = std::chrono::steady_clock::now();
            tq.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            std::fill(yq.begin(), yq.end(), 0.f);
        }
        // recompute deterministically for the error metric
        {
            std::vector<float> yy(sp.n_out, 0.f);
            for (int b = 0; b < nb; ++b) {
                const uint8_t *slab = &Q[(size_t)b * rows * 18];
                const float *xb = x.data() + b * DB;
                for (int o = 0; o < sp.n_out; ++o) yy[o] += dot_iq4(slab + (size_t)o * 18, xb);
            }
            yq = yy;
        }
        double rel = 0, nrm = 0;
        int nnan = 0;
        double my = 0, mqx = 0;
        for (int o = 0; o < sp.n_out; ++o) {
            if (!std::isfinite(y[o]) || !std::isfinite(yq[o])) { ++nnan; continue; }
            rel += (yq[o] - y[o]) * (yq[o] - y[o]);
            nrm += y[o] * y[o];
            my = std::max(my, (double)std::fabs(y[o]));
            mqx = std::max(mqx, (double)std::fabs(yq[o]));
        }
        fprintf(stderr, "  %s nan=%d max|y|=%.4g max|yq|=%.4g\n", sp.tag, nnan, my, mqx);
        std::sort(tb.begin(), tb.end());
        std::sort(tq.begin(), tq.end());
        double mb = tb[reps / 2], mq = tq[reps / 2];
        printf("%s,%d,%d,bf16,%.4f,%.2f,%.2f\n", sp.tag, sp.n_out, sp.d, mb / pool,
               bytes_bf16 * pool / mb / 1e6, bytes_bf16 / 1048576.0);
        printf("%s,%d,%d,iq4,%.4f,%.2f,%.2f\n", sp.tag, sp.n_out, sp.d, mq / pool,
               bytes_iq4 * pool / mq / 1e6, bytes_iq4 / 1048576.0);
        fprintf(stderr, "  %s rel_l2(iq4 vs bf16)=%.4f  speedup=%.2fx\n", sp.tag,
                std::sqrt(rel / nrm), mb / mq);
        W.clear(); W.shrink_to_fit();
        Q.clear(); Q.shrink_to_fit();
    }
    return 0;
}
