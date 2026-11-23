# Phase 5 — verification gate and report build

## Gate result (`scripts/verify_all.py` → `results/verify_all.json`)

| check | measured | verdict |
|---|---|---|
| π_ffn isolation, fp32, real layer-0 weights | rel L2 **3.06e-7** (max|Δ| on an output of scale 7.4e-3) | pass |
| π_ffn identity-permutation control | **0.0** (bitwise) | pass |
| π_ffn one-sided control (permute h only) | rel L2 **1.26** | fires |
| π_conv isolation, fp32, real layer-0 weights | rel L2 **3.39e-7** | pass |
| π_conv identity control | **0.0** (bitwise) | pass |
| π_conv one-sided control (permute conv taps only) | rel L2 **1.37** | fires |
| scene-cut drift, fully flushed clips (V4, V6) | max|Δ| **0.0**, KL **0.0** | pass |
| peak anonymous RSS during the gate | **409.8 MB** (< 1536 MB) | pass |
| `all_pass` | **True** | pass |

The 10^6 separation between the exact case and the one-sided control is the important part:
an exact-looking number can also come from a test that does nothing.

Two gate-design fixes were needed and are worth recording:
1. the first version measured `max|Δlogit|` on unscaled `N(0,1)` activations, so the fp32
   reordering error of a 7680-term dot product read as 42.9 on an output scale of 900 —
   meaningless. The gate now reports **relative L2** plus the output scale;
2. the first scene-cut assertion also required `KL == 0` for *all* clips, but V1–V3 legitimately
   drop 55–60% of tokens (KL 1e-6). The guarantee is per **flush frame**, so the gate now
   checks `scene_cut_maxabs_flush == 0.0` for all clips and `KL == 0.0` for the fully flushed
   ones.

## Report build

`/tmp/tectonic -X compile CM-LFM3.tex` → **CM-LFM3.pdf, 18 pages** (no LaTeX toolchain was
installed; tectonic 0.15.0 was fetched, it builds the document from source in a temp dir).
Figures are generated from the measured JSONs by `scripts/make_figures.py` (no synthetic
data): `figs/fig_ffn_ppl.pdf`, `figs/fig_conv_ppl.pdf`, `figs/fig_energy.pdf`,
`figs/fig_video.pdf`. Tables 9–11 and 13 are `\input` from generated fragments
(`results/tab_ablation_{ffn,conv,both}.tex`, `results/tab_phase3.tex`) so the report cannot
drift from the JSON.

## Compliance checklist against CLAUDE.md

| rule | status |
|---|---|
| no full checkpoint in anonymous RAM | ok — mmap/`safe_open`, RssAnon ≤ 1.32 GB in the heaviest run, 410 MB in the gate |
| zero-copy streaming, per-layer processing | ok — `safe_open(...).get_tensor`, calibration accumulators freed per group |
| never load both models in one process | ok — only LFM3-VL-2B was ever loaded |
| `chmod 444` + SHA-256 | ok — `artifacts/checksums.sha256`, `.safetensors` is `-r--r--r--` |
| `OMP_WAIT_POLICY=passive`, 16 cores, bind close | ok — all pinned CPU runs |
| no runtime `torch.quantile` in the deployment path | ok — `ours_pt` is flagged analysis-only |
| no static-norm gating; no row-major sub-64B skipping | ok — runtime block energy only, block-major slabs |
| video masks with 3×3 majority pre-dilation + ŝ>0.5 flush | ok — plus a global-motion gate |
| every number measured, no mock timers | ok — `steady_clock` / `cudaEvent` / `__rdtsc` only |

## Open items (honest)

1. The Triton/CUDA block-sparse prefill kernel is **not** implemented; the GPU numbers are
   masked-dense gating, so the block-sparsity factor is measured on CPU only. This is the
   single largest gap against the original plan.
2. LFM3-1.7B is not available locally (the download was started and is incomplete at
   `/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-1.7B`); all dual-model comparisons were
   therefore dropped by the user's own instruction ("另外一个提及的模型不重要，忽略它").
3. Vision-tower conv layers (stateless, k=3) were profiled but not kernel-ised.
