# CM-LFM3 — Technical Report 3 of the CM-Series

Inference-time sparsification of **LFM3-VL-2B**: isomorphic channel permutation,
cache-line aligned block sparsity, and spatio-temporal token skipping for Liquid
Foundation Models.

* Report: `CM-LFM3.tex` → `CM-LFM3.pdf` (24 pages)
* Phase reports: `results/PHASE{1..5}_REPORT.md`
* Measurements: `results/*.json`, `results/*.csv`, `results/tab_*.tex`
* Kernels and scripts: `src/`, `scripts/`
* Weight integrity: `artifacts/checksums.sha256`

## Contents

| Phase | What it establishes | Entry point |
|---|---|---|
| 1 | Anatomy of LFM3-VL-2B (13 `LFM3Conv` / 11 GQA layers, 59.0% of bytes in the SwiGLU FFN), measured 44 GB/s DRAM roof, zero-copy streaming under 1.5 GB anonymous RSS, dense quality/latency baselines | `scripts/arch_anatomy.py`, `scripts/phase1_*.py` |
| 2 | Dual isomorphism (SwiGLU and `LFM3Conv`) to 3.1e-7 / 3.4e-7 fp32 rel L2; block construction; per-token RMS threshold rule; the six-method ablation | `src/lfm3_sparse.py`, `scripts/phase2_*.py` |
| 3 | Fused AVX-512 `LFM3Conv` kernel, block-major 4.5-bit IQ4_NL, block skipping, Triton block-sparse GEMV, decode ledger | `src/p3_conv_kernel.cpp`, `src/p3_kernels.cpp`, `src/p3_gqa.cpp`, `scripts/phase3_*.py` |
| 4 | Five-stage spatio-temporal `P2` token skipping with bit-identical scene-cut fallback | `scripts/phase4_*.py` |
| 5 | Verification gate and report build | `scripts/verify_all.py`, `scripts/make_figures.py` |

## Headline numbers

* Isomorphism at τ=0: rel L2 **3.06e-7** (FFN) / **3.39e-7** (`LFM3Conv`), identity control exactly 0.0, one-sided control 1.26 / 1.37.
* FFN gating at τ=0.5: ΔPPL **+2.49** (per-token RMS rule) versus **+20.81** (native top-K) and **+10.28** (CATS).
* `LFM3Conv` gating is lossless: mean ΔPPL **−0.17** over five budgets.
* Kernels: **1.48×** fusion at T=64; **2.58×** (FFN) and **4.07×** (conv) per layer at T=1 decode; Triton block-sparse GEMV **3.15×** on the RTX 5070.
* CPU decode ledger: 105.07 → 31.70 ms/token composed (**3.31×**; pessimistic component-scaled bound 2.26×), floor 30.9 ms for 1130 MB of remaining weight traffic.
* Video token skipping: 55.5–59.5% of prefill tokens removed for 1.75–1.84× prefill speedup, KL ≤ 1e-6, and `max|Δ| = 0.0` on scene cuts and camera pans.

## Reproduction

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch transformers==5.19.0 accelerate safetensors datasets matplotlib triton
export OMP_WAIT_POLICY=passive OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=16
python scripts/phase1_baseline.py --model <path-to-LFM3-VL-2B> ...
python scripts/verify_all.py
tectonic -X compile CM-LFM3.tex
```

Model weights are not part of this repository; `artifacts/checksums.sha256` records the
SHA-256 of every file used. Calibration tensors (`results/*.pt`), weight dumps
(`artifacts/p3/`) and upstream dataset/video frames are excluded — they are either large
binary artifacts or third-party content that is not ours to redistribute.
