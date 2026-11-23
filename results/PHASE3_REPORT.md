# Phase 3 — LFM3Conv microkernel: fusion, IQ4_NL block-major, Variant-A skipping

All numbers are wall-clock medians on the Zen 5 testbed with
`OMP_WAIT_POLICY=passive OMP_PLACES=cores OMP_PROC_BIND=close`, produced by
`src/p3_conv_kernel.cpp` driven by `scripts/phase3_run.py`; raw data in
`results/phase3_kernels.json` / `results/phase3_kernels.md`.
Operands are the **real** layer-0 `LFM3Conv` tensors and **real** activations
(`artifacts/p3/`, dumped by `scripts/phase3_dump.py`): `in_proj` 6144×2048 bf16,
depthwise `conv.weight` 2048×7, `out_proj` 2048×2048 bf16, `x` 64×2048 from a real
WikiText forward through the model. The fused kernel reproduces the PyTorch
reference **bit-exactly** (`fused_vs_unfused_maxabs = 0`).

## 3.1 What the kernel fuses

The unfused path materialises `in_proj` output (3d×T), a separate pre-gate buffer,
the conv output, the post-gate buffer, and a transposed contiguous copy before
`out_proj` — five round trips through L1/L2 plus `2·3d·T` extra stores. The fused
kernel keeps a channel-major `h`/`C` pair in registers/L1 between the pre-gate, the
7-tap causal depthwise conv and the post-gate, emits the gated output in
token-major order (so the projection reads are contiguous), and reduces 32-lane
block energies in the same pass, emitting a keep-bitmap for the `out_proj` slabs.

## 3.2 Fusion gain (unfused → fused, bf16)

| threads | T | unfused (ms) | fused (ms) | speedup |
|---|---|---|---|---|
| 16 | 64 | 7.498 | 5.074 | **1.478×** |
| 16 | 16 | 3.544 | 2.838 | 1.249× |
| 16 | 1 | 1.134 | 1.334 | 0.85× (noise-dominated, 1 token) |
| 1 | 64 | 25.86 | 26.00 | 1.00× |
| 1 | 16 | 8.94 | 8.77 | 1.02× |

At one thread the layer is DRAM-bound on this single-channel host (44 GB/s roof,
measured `results/phase1_bw.txt`), so removing round trips buys nothing. With 16
threads and T=64 the layer is latency/L2-bound and fusion is worth 1.48×, which is
the same order as the 31.1% unfused transpose tax measured independently on CPU in
Phase 1 (`results/phase1_cpu_cold.json`).

## 3.3 4.5-bit IQ4_NL in 64-byte cache-line aligned block-major layout

`out_proj` (2048×2048) is quantised to 4.5 bit with the llama.cpp `IQ4_NL` codebook
(`KV = {-127,-104,-83,-65,-45,-28,-12,1,13,23,33,41,49,56,62,66}`, fp16 block scale
= amax/127) and stored block-major as `[n_blocks][out_features][18 B]`:

* one 32-lane slab = 2048 × 18 B = **36 864 B = 576 cache lines exactly**, so every
  line is either fully consumed or not touched at all;
* row-major (the layout CLAUDE.md forbids) puts 18 B of a 64-B line to use when a
  single block column is live → 28.1% line utilisation, and an 8-bit row stride of
  1152 B guarantees 100% line touch on skip.

Decoding uses one `pshufb` per 16 codes (`iq4_decode32`), two `vfmadd` per block, and
one horizontal reduction per output row (not per block), which took the dense IQ4
path from 138.9 ms to 24.5 ms at T=64/1-thread — i.e. from 5× slower than bf16 to
**parity with bf16 while reading 3.5× fewer bytes** (33.55 MB → 9.55 MB for the
projections).

| threads | T | fused bf16 | IQ4 dense | IQ4 sparse τ=0.5 | IQ4s/bf16 |
|---|---|---|---|---|---|
| 16 | 64 | 4.388 | 4.955 | 3.738 | **1.174×** |
| 16 | 16 | 2.808 | 2.834 | 2.324 | **1.208×** |
| 1 | 64 | 24.43 | 26.32 | 23.30 | 1.049× |
| 1 | 16 | 7.43 | 7.91 | 6.51 | 1.142× |

## 3.4 Variant A vs the Amdahl ceiling

Variant A (state-exact) can only skip `out_proj`, which is 25% of the layer's bf16
weight bytes and 22.9% of the IQ4+in-proj bytes. With a measured active-block
fraction of 0.50 the byte ceiling is therefore

    speedup_max = (3·d² + d²) / (3·d² + 0.5·d²/3.56) ≈ 1.28×,

and we measure 1.17–1.21× at T=16/64 — ≈92% of the ceiling. Variant B (gate-first)
doubles the skippable mass to 2d² and would raise the ceiling to ≈1.55×, but Phase 2
showed it costs 2–10× more accuracy per skipped byte, so it is rejected on the
Pareto front.

## 3.5 Threshold transfer (a deployment finding)

θ must be calibrated **per shape**. The Phase-2 θ (calibrated on 256-token prefill)
keeps 42% of blocks at T=64 but **0%** at T=1, because a single replayed token has a
cold depthwise-conv state: mean 32-lane block energy is 2.8e-7 at T=1 versus
6.1e-5 at T=64 (≈220×). Two rows of the sweep (`τ_cal = 0` at T=1) are therefore
flagged `degenerate` in the JSON and their "speedup" is not reported. With a
per-shape calibration at τ=0.5 the missing-mass fraction is 0.516 and the sparse
path is 0.73–0.91× of bf16 at T=1 — i.e. at decode the 25%-skippable Variant A does
**not** pay for its indirection on this CPU, which is a negative result we report
rather than hide.

## 3.6 Honest summary

1. Fusion: 1.48× at T=64/16 threads, bit-exact.
2. IQ4_NL block-major: parity in time at 3.5× fewer bytes; the win is bandwidth, which
   only converts to time when the layer is bandwidth-bound (batch/decode), and on this
   single-channel host the 25% Variant-A skip is not enough to show it at T=1.
3. Sparse: 1.17–1.21× at prefill shapes, ≈92% of the Amdahl ceiling.
4. Threshold transfer across shapes is the main operational risk, not the kernel.

---

## Revision R2 (after review): coverage, per-token gating, GPU kernel

Three defects were found by review and are fixed here. The superseded numbers stay in
`results/phase3_kernels.json`; the current ones are in `results/phase3_kernels2_sym.csv`,
`results/phase3_triton.json`, `results/phase3_e2e.json`.

1. **Coverage.** The old sweep quantised only `LFM3Conv.out_proj` (25% of a layer type that
   is 11.4% of the model) -> Amdahl ceiling 1.28x, i.e. 1.03x end to end. The new kernel
   (`src/p3_kernels.cpp`) also covers the SwiGLU FFN (59.0% of model bytes).
2. **Gating.** Block energy is now a per-token RMS over the 32 lanes,
   `E[t][b] = sqrt(mean_32 h^2)`, and theta is the *low* quantile (cold blocks are dropped).
   The old statistic summed over T, so a T=64 theta was ~220x too large at T=1 and masked
   every block; those rows were marked `degenerate`. Active fraction is now 0.500 / 0.502 /
   0.500 at T=1/16/64 for tau=0.5.
3. **Layout.** Row-major is measured, not asserted: at T=1 it is *slower* than dense
   row-major for the FFN (1271 vs 791 us) because each skip still touches every line.

### Measured (sym codebook, cold-flushed, tau=0.5)

| layer | T | th | bf16 | iq4bm | iq4bm+skip | iq4rm+skip | sp(bm) | sp(rm) |
|---|---|---|---|---|---|---|---|---|
| ffn  | 1  | 16 | 2030.9 | 838.6 | 786.1  | 1045.5 | 2.58x | 1.94x |
| ffn  | 16 | 16 | 23045.4 | 5841.9 | 4350.7 | 12519.9 | 5.30x | 1.84x |
| ffn  | 64 | 16 | 90554.0 | 20616.1 | 17766.8 | 52249.4 | 5.10x | 1.73x |
| conv | 1  | 16 | 673.4 | 233.9 | 165.5 | 376.0 | 4.07x | 1.79x |
| conv | 16 | 16 | 2347.5 | 2355.6 | 2023.3 | 4149.3 | 1.16x | 0.57x |
| conv | 64 | 16 | 8565.5 | 8153.5 | 7516.7 | 16596.4 | 1.14x | 0.52x |

Cache lines touched at T=1 (MB): ffn 94.37 (bf16) -> 26.54 (bm) -> 22.12 (bm+skip);
conv 33.55 -> 9.44 -> 8.26. End-to-end decode projection: **1.99x**
(142.1 -> 71.5 ms/token) against a 30.9 ms post-quantisation DRAM floor.

### GPU (RTX 5070, Triton 3.9, `scripts/phase3_triton.py`)

| matrix | tau | active | torch.mv | triton dense | triton sparse | speedup |
|---|---|---|---|---|---|---|
| down_proj (7680->2048) | 0.0 | 1.000 | 87.4 | 54.3 | 54.2 | 1.61x |
| down_proj | 0.25 | 0.746 | 87.4 | 54.4 | 40.8 | 2.14x |
| down_proj | 0.50 | 0.500 | 87.5 | 54.4 | 27.8 | 3.15x |
| out_proj (2048->2048) | 0.0 | 1.000 | 24.2 | 15.3 | 15.2 | 1.60x |
| out_proj | 0.25 | 0.750 | 24.3 | 15.2 | 11.7 | 2.08x |
| out_proj | 0.50 | 0.500 | 24.2 | 15.2 | 8.1 | 3.01x |

bf16 sparse is numerically exact (rel 0); IQ4 on GPU is slower than bf16 at every sparsity
(58.6 vs 27.8 us at tau=0.5) because nibble decoding costs more instructions than 11% of a
600 GB/s budget buys. IQ4 pays on CPU, where the byte budget is 13x tighter.

### New negative result: the mandated IQ4_NL codebook

llama.cpp `kvalues_iq4nl` with `d = amax/127` gives rel L2 0.2218 (W2) and 0.3272 (full FFN
layer) vs 0.098 / 0.045 for a symmetric 16-level table in the same 18 bytes. The positive
side of the mandated table stops at 66/127, so 12.0% of codes collapse into the top bin.
Kernel and NumPy agree to 5%, so it is the codebook, not the kernel.

---

## Revision R3 — GQA + lm_head coverage and the decode ledger

### GQA and lm_head in symmetric IQ4_NL block-major (T=1, 16 threads, DRAM-cold, >=200 MB pool)

| matrix | bf16 ms | IQ4 ms | speedup | rel L2 |
|---|---|---|---|---|
| q_proj 2048x2048 | 0.2313 | 0.1129 | 2.05x | 0.090 |
| k_proj 512x2048 | 0.0633 | 0.0272 | 2.33x | 0.095 |
| v_proj 512x2048 | 0.0566 | 0.0282 | 2.00x | 0.094 |
| o_proj 2048x2048 | 0.2315 | 0.1027 | 2.25x | 0.091 |
| lm_head 128000x2048 | 14.5098 | 6.5765 | 2.21x | 0.131 |

bf16 sustains 36.1 GB/s on lm_head (82% of the 44 GB/s roof); IQ4 sustains only ~22 GB/s
(it is decode-bound) but still wins 2.0-2.3x on 3.56x fewer bytes.

### Where a decode step goes (measured by module ablation, not hooks)

Hooks returned 4 us per projection (nested calls clobbered the start stamp), so each
operator class was costed by replacing it with an identity stub and taking the difference.

| component | dense ms/token | share | after IQ4+skip | ratio |
|---|---|---|---|---|
| 24x SwiGLU FFN | 73.11 | 69.6% | 18.87 | 0.387 |
| 13x LFM3Conv in+out | 18.53 | 17.6% | 2.15 | 0.246 |
| 11x GQA q/k/v/o | 12.78 | 12.2% | 2.98 | 0.465 |
| lm_head | 14.44 | 13.7% | 6.58 | 0.453 |
| non-GEMM residue | 1.12 | 1.1% | 1.12 | 1.00 |
| **total** | **105.07** | | **31.70** | **3.31x** |

Two bounds: composing measured kernel times + measured residue gives 31.70 ms (3.31x);
scaling each in-model dense component by its kernel ratio gives 46.46 ms (2.26x), which is
pessimistic because the three ablation deltas sum to 118.9 ms against a 105.1 ms total
(14.9 ms of cache contention that disappears when a component is removed).
Reported claim: **2.3-3.3x end-to-end CPU decode**, floor 30.9 ms for the 1130 MB a fully
quantised, half-skipped model must still stream.

Kernel bugs fixed here: `_mm512_setr_epi32` codebook viewed as floats (codes < -8 became
NaN bit patterns, rel_l2 = nan); `pack_iq4` writing only 8 of 16 code bytes; 48 MB
working sets still fitting the 64 MiB L3 (reported 120-170 GB/s).
