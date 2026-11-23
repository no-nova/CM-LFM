// Phase 1 memory-roof probe for the CM-LFM3 report.
// Measures single-socket STREAM triad / read / copy / write with AVX-512,
// 1 and N threads, and 8-byte-per-element streaming stores off.
// Build: g++ -O3 -march=native -mavx512f -mavx512bw -mavx512vl -mfma -fopenmp -std=c++20 bw_probe.cpp -o bw_probe
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  size_t mb = argc > 1 ? std::stoul(argv[1]) : 1024;      // MiB per buffer
  int reps = argc > 2 ? std::stoi(argv[2]) : 5;
  size_t n = mb * 1024 * 1024 / sizeof(float);
  size_t n8 = n & ~7UL;
  std::vector<float> a(n8, 1.0f), b(n8, 2.0f), c(n8, 0.0f);
  std::printf("# buffers %zu MiB x3, n=%zu (%.0f MiB/buffer), reps=%d\n", mb, n8,
              n8 * 4.0 / 1048576.0, reps);
  std::printf("%-6s %-8s %10s %10s %10s\n", "thr", "kernel", "GB/s", "MiB/s", "ms");
  for (int th : {1, 2, 4, 8, 16, 32}) {
    omp_set_num_threads(th);
    auto run = [&](const char* name, void (*k)(float*, float*, float*, size_t), double bytes) {
      // warm
      k(a.data(), b.data(), c.data(), n8);
      double best = 1e9;
      for (int r = 0; r < reps; ++r) {
        double t0 = now();
        k(a.data(), b.data(), c.data(), n8);
        double t1 = now();
        best = std::min(best, t1 - t0);
      }
      double gbs = bytes * n8 / best / 1e9;
      std::printf("%-6d %-8s %10.2f %10.0f %10.3f\n", th, name, gbs, gbs * 1024, best * 1e3);
    };
    run("read", [](float* a, float* b, float* c, size_t n) {
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        __m256 w = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(v, w));
      }
    }, 24.0);   // 3 reads + ... 2 reads 1 write = 12 B/elem
    run("write", [](float* a, float* b, float* c, size_t n) {
      __m512 z = _mm512_set1_ps(3.0f);
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i += 16) _mm512_storeu_ps(c + i, z);
    }, 4.0);
    run("copy", [](float* a, float* b, float* c, size_t n) {
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i += 16) {
        __m512 v = _mm512_loadu_ps(a + i);
        _mm512_storeu_ps(c + i, v);
      }
    }, 8.0);
    run("triad", [](float* a, float* b, float* c, size_t n) {
      __m512 s = _mm512_set1_ps(2.0f);
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i += 16) {
        __m512 x = _mm512_loadu_ps(b + i);
        __m512 y = _mm512_loadu_ps(c + i);
        _mm512_storeu_ps(a + i, _mm512_fmadd_ps(s, x, y));
      }
    }, 12.0);
  }
  return 0;
}
