# Phase 3 (revised): block-major IQ4_NL with per-token RMS gating

Microseconds per layer call, cold-flushed, sym codebook. Speedups are against dense bf16 at the same (kind, T, threads).

| layer | T | th | bf16 | iq4bm | iq4bm+skip | iq4rm | iq4rm+skip | sp(bm+skip) | sp(rm+skip) |
|---|---|---|---|---|---|---|---|---|---|
| ffn | 1 | 1 | 3674.7 | 2174.5 | 2654.2 | 2304.5 | 2133.6 | 1.38x | 1.72x |
| ffn | 16 | 1 | 49529.5 | 27127.5 | 22426.6 | 30796.1 | 29074.5 | 2.21x | 1.70x |
| ffn | 64 | 1 | 197364.2 | 88843.2 | 84613.2 | 122973.2 | 116155.1 | 2.33x | 1.70x |
| ffn | 1 | 16 | 2030.9 | 838.6 | 786.1 | 790.5 | 1045.5 | 2.58x | 1.94x |
| ffn | 16 | 16 | 23045.4 | 5841.9 | 4350.7 | 5396.8 | 12519.9 | 5.30x | 1.84x |
| ffn | 64 | 16 | 90554.0 | 20616.1 | 17766.8 | 22581.9 | 52249.4 | 5.10x | 1.73x |
| conv | 1 | 1 | 1253.0 | 783.4 | 700.0 | 653.5 | 644.2 | 1.79x | 1.95x |
| conv | 16 | 1 | 8984.4 | 7061.5 | 6318.2 | 10366.9 | 10024.2 | 1.42x | 0.90x |
| conv | 64 | 1 | 34112.3 | 28349.1 | 25470.4 | 41482.6 | 40426.1 | 1.34x | 0.84x |
| conv | 1 | 16 | 673.4 | 233.9 | 165.5 | 311.0 | 376.0 | 4.07x | 1.79x |
| conv | 16 | 16 | 2347.5 | 2355.6 | 2023.3 | 2515.0 | 4149.3 | 1.16x | 0.57x |
| conv | 64 | 16 | 8565.5 | 8153.5 | 7516.7 | 8702.9 | 16596.4 | 1.14x | 0.52x |

Cache lines touched (MB) at T=1, 16 threads, tau=0.50:

| layer | bf16 | iq4 bm | iq4 bm+skip | iq4 rm+skip |
|---|---|---|---|---|
| ffn | 94.37 | 26.54 | 22.12 | 28.97 |
| conv | 33.55 | 9.44 | 8.26 | 11.01 |

## End-to-end decode projection (CPU, 16 threads, T=1)

- FFN 59.04% of bytes at 2.58$\times$ -> 0.2536 time units
- LFM3Conv 11.38% at 4.07$\times$ -> 0.0310
- GQA 6.03% unquantised -> 0.0669; lm_head/embed 13.67% unquantised -> 0.1517
- total 0.5032 units -> **1.99$\times$ end-to-end** (142.1 ms -> 71.5 ms per token; the post-quantisation DRAM floor is 30.9 ms for 1359 MB, so the layer kernels run at 43% of the floor)
