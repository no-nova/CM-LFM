# CM-LFM3 — Phase 1 report: architectural anatomy & dense baselines
Model: `LiquidAI/LFM3-VL-2B` (local: `/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B/`, `model.safetensors` = 3,836,417,920 B = 3.573 GiB, bf16, 424 tensors)
SHA-256 (all files) in `artifacts/checksums.sha256`; `model.safetensors` is `chmod 444`.
Testbed: Ryzen 9 9955HX (16C/32T, AVX-512F/BW/DQ/VL/VNNI/BF16), 64 MiB L3, one usable DDR5 channel (44 GB/s read, measured), RTX 5070 Laptop 8 GB GDDR7, torch 2.15.0, transformers 5.19.0.

---

## 1. Layer topology (read from the checkpoint, `scripts/phase1_arch_report.py`)

| property | text backbone | vision tower |
|---|---|---|
| hidden size d | 2048 | 1024 |
| FFN intermediate d_ff | 7680 (SwiGLU) | 2560 |
| layers | 24 | 14 |
| layer types | **LFM3Conv ×13, full-attention ×6, sliding-attention ×5** | LFM3Conv ×8, full-attention ×6 |
| heads | 16 × 128 (q/o), **4 KV heads** (4:1 GQA) | 16 / 8 KV heads |
| conv kernel k | 7 (`conv_L_cache`), depthwise, causal | 3 |
| sliding window | 256 | – |
| vocab / tying | 128,000 / tied (no `lm_head` tensor) | – |

* `LFM3Conv` = `Lfm3ShortConv`: `in_proj` (d→3d, no bias) → split into **B, C, x** → `h = B⊙x` → depthwise causal `conv1d(k=7, groups=d)` → `y = C⊙h` → `out_proj` (d→d). Only channel-wise ops between `in_proj` and `out_proj` — this is the structural fact that makes π_conv isomorphic.
* Attention layers additionally carry a per-head sigmoid `gate_proj` (d→16); decoder layers use sandwich norms (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`), all `Lfm3RMSNorm`.
* Multimodal projector: `linear_1` 2048→4096 + GELU + LayerNorm + `linear_2` 4096→2048 (48.0 MiB, 1.31%).
* Vision patch path: patch 16, temporal patch 2, spatial merge 2 (one language token per 2×2 merged patch block).

## 2. Parameter / byte budget and arithmetic intensity

| component | bf16 MiB | share of 3658.6 MiB |
|---|---|---|
| text FFN (24 × SwiGLU w1,w3,w2) | 2160.0 | **59.04%** |
| text LFM3Conv (13 × in_proj/out_proj/dwconv) | 416.4 | 11.38% |
| vision tower (8 conv + 6 attn + 14 FFN) | 313.2 | 8.56% |
| text GQA attention (11 ×) | 220.7 | 6.03% |
| embed_tokens (tied lm_head) | 500.0 | 13.67% |
| multimodal projector | 48.0 | 1.31% |

Parameter formulas (d = 2048, d_ff = 7680): conv layer = 3d² + d² + kd = 16,791,552; attention layer = d·n_h·h + 2·d·n_kv·h + d·n_h·h + d·n_h + 2h = 10,518,528; FFN = 3·d·d_ff = 47,185,920.

**α_l (FLOPs / byte moved, bf16, weights streamed once):** all six projection families sit at **α = 1.0 FLOP/B at T = 1** (weight-stationary GEMV, hard memory bound), rising to α ≈ 51–63 at T = 64 (q/o 60.2, k/v 51.2, w1/w3 62.9, w2 62.9, conv in_proj 62.7, conv out_proj 60.2). Consequence: decode on this host is 100% bandwidth-limited; the roofline ceiling is 3.58 GB / 44 GB/s = **81 ms/token on CPU** and 3.58 GB / 672 GB/s ≈ 5.3 ms/token on GPU.

## 3. Memory roof (measured, `scripts/bw_probe.cpp`, 1 GiB × 3 buffers, `OMP_WAIT_POLICY=passive`)

| threads | read | write | copy | triad |
|---|---|---|---|---|
| 1 | 32.8 | 9.5 | 14.5 | 16.8 GB/s |
| 16 | 41.1 | 12.6 | 17.3 | 20.7 GB/s |
| 32 | **44.0** | 12.7 | 17.7 | 20.9 GB/s |

One DDR5 channel is populated → 44 GB/s read roof (not 208 GB/s). 16 vs 32 threads is within noise for streaming; for small ops 32 threads is *worse* (hyperthread contention).

## 4. GPU (bf16) dense baselines

| metric | value |
|---|---|
| weights in VRAM / load time | 3.573 GiB / 2.0 s |
| prefill T=16 | 12.10 ms (1,322.8 tok/s) |
| prefill T=64 | 12.15 ms (5,265.8 tok/s) |
| prefill T=256 | 23.38 ms (10,947.7 tok/s) |
| prefill T=1024 | 88.58 ms (11,559.8 tok/s) |
| decode (KV cache) | **11.42 ms/token = 87.6 tok/s** |
| VL prefill, 1 image, 91 tokens | 19.05 ms |
| peak VRAM (T=1024) | 7.043 GiB (T≥2048 OOM → chunk ≤1024, `expandable_segments:True`, `@torch.inference_mode`) |

Effective decode bandwidth = 3.58 GB / 11.42 ms = **314 GB/s = 47% of the 672 GB/s GDDR7 roof** → 2.1× headroom for Phase 3.

Op distribution (1024-token prefill, `results/phase1_ops.csv`; total 171 ms is profiler-inflated, use shares):
GEMM 76.8% · elementwise mul (gates, silu·mul) 8.0% · FMHA 4.2% · **depthwise conv1d 2.6% (358 launches)** · layout copies 1.7% · rsqrt 1.6% · silu/sigmoid 1.2% · reductions 1.0%. 698 elementwise-mul launches per prefill is the fusion head-room Phase 3 targets.

## 5. CPU dense baselines (bf16 weights via zero-copy `mmap`, 16 threads unless stated)

| metric | value |
|---|---|
| checkpoint load (meta-init + `assign=True` + rebuild rotary buffers) | 1.53 s, RssAnon 473 MB |
| prefill T=64 | 624.1 ms |
| decode | **142.1 ms/token** (roof 81 ms → 57% of roof) |
| cold GEMV bf16, 1 thread | 20.4–23.5 GB/s (46–53% of the 44 GB/s roof) |
| cold GEMV bf16, 16 threads | 19.1–25.9 GB/s (no gain → memory bound) |
| cold GEMV fp32, 1 thread | 17.7–24.0 GB/s |
| fp32 T=64 FFN GEMM | 317.9 GFLOPS |
| CPU vs GPU, identical bf16 weights | hidden-state RMSE 0.1045, rel L2 3.9% |

**LFM3Conv unfused tax** (reference path vs transpose-free variant, `results/phase1_cpu_cold.json`): T=1 bf16 **28.6%** (1 thread) / **31.1%** (16 threads); fp32 14.4% / 8.8%; at T=16–64 it vanishes (<1%) because the GEMM dominates. Decode-time fusion is therefore worth ~30% of the conv-block cost on CPU.

## 6. Quality baselines

WikiText-2 test (294,512 tokens, `results/phase1_ppl.json`):

| protocol | PPL | bits/token | tokens scored |
|---|---|---|---|
| P1 sliding-window 1024/512 (full test set) | 68.61 | 6.100 | 294,399 |
| P2 continuous KV cache, first 16k tokens | 63.84 | 5.996 | 16,096 |
| P3 chat-conditioned (assistant span, 32×256) | **42.68** | 5.415 | 7,296 |

Window-restart penalty (NLL per position, 512-token windows): pos 0–31 = 7.64, pos 32–255 = 5.49, pos 256–511 = 4.57 → the 68.6 vs 63.8 gap is entirely warm-up, consistent with `sliding_window = 256`.

Caveats recorded honestly: raw-text PPL is off-distribution for this chat/thinking model (chat-template PPL is 42.7 and the model is otherwise demonstrably intact — image MC accuracy 100%, correct chat answers). Baseline PPL is used only as the *reference point* for relative degradation in Phase 2, and every ΔPPL is measured against the same-kernel-path dense model.

Vision-language baselines (`results/phase1_vqa.json`, position-balanced, both option orders):
* image multiple-choice VQA **100%** (10 images × 2 orders).
* pairwise "which clip moves more" **58.3%** (7/12) → near chance; binary motion questions are answered "yes" unconditionally (12/12 on an earlier run, and the frozen-clip control also said "yes"), so Phase 4's VideoMME-style evaluation must be pairwise + frozen controls, never yes/no.
* synthetic scene-cut detection 25%, frozen-clip control 66.7% → the model cannot be used as the metric; Phase 4 must score token-skipping by **feature drift**, not by asking the model.
* VL latency ≈ 2.7 s per question (2 clips, ~260 tokens).

## 7. Numeric-path findings that constrain Phase 2–5 methodology

1. Kernel-path sensitivity: sdpa vs eager attention gives logit RMSE 0.079 / Top-1 0.959 with identical weights. Any Δ inside the model must therefore be measured **within one kernel path**.
2. bf16 tensor-core reordering: a real channel permutation changes end-to-end bf16 logits by RMSE ≈ 0.098 even though the operator is mathematically identical (identity permutation and plain reload both give exactly 0.0; a bf16 K-permuted GEMV at operator level is exact to 0.0–1.9e-4 relative). ⇒ **all Phase-2 quality numbers use the permuted-dense model as the τ=0 reference**, and the τ=0 isomorphism gate is certified in fp32 (§8) and per layer in bf16 (≤1.2e-4 abs at output scale 0.1).
3. Memory-safety rule observed everywhere: weights via `safetensors` mmap / `low_cpu_mem_usage`, never `torch.load` of the full checkpoint; single model per process; peak **RssAnon 1.32 GB** in the heaviest (PPL) run, 1.59 GB in the video VQA run (12 clips resident); `VmRSS` 4.8 GB is dominated by reclaimable mmap page cache and is not anonymous memory.

## 8. Phase 2 step 1 (already completed): τ = 0 isomorphism verified

`results/phase2_isomorphism.json`, `phase2_isomorphism_fp32.json`, `phase2_iso_diag.json`, `phase2_iso_layer_probe.json`

| check | max abs | rel L2 |
|---|---|---|
| FFN π, float64, operator | 2.7e-14 | 3.7e-15 |
| FFN π, float32, operator | 1.5e-6 | 3.8e-7 |
| LFM3Conv π, float64, operator | 1.2e-15 | 1.9e-15 |
| LFM3Conv π, float32, operator | 1.4e-7 | 3.5e-7 |
| negative control (gate permuted, `out_proj` not) | 0.53–0.58 | 1.0 |
| **whole model fp32, 37 layers, π_ffn** | 1.04e-4 | **2.29e-6** |
| **whole model fp32, 37 layers, π_conv** | 6.10e-5 | **2.37e-6** |
| restore (inverse permutation) | 0.0 (bitwise) | 0 |
| identity permutation, bf16 whole model | 0.0 (bitwise) | 0 |

Criterion satisfied: rel L2 ≤ 1e-5 for both π, with the negative control proving the test is sensitive.
