# Phase 4 — Five-stage spatio-temporal token skipping (P2) on LFM3-VL-2B

Implementation: `scripts/phase4_video.py`; thresholds calibrated once on a known-static
clip by `scripts/phase4_calibrate.py` → `results/phase4_calib.json`
(`TAU_ABS = 0.03388`, i.e. the 95th percentile of 32×32 block frame differences on
`V1_static_surveillance`; global-motion gate `GAMMA = 1.5` × `NOISE_MEAN = 0.0170`).
Raw results: `results/phase4_video.json`. Real 224×224 frames from
`data/real_videos/frames_224`, 8 frames per clip, real model, real GPU timing
(median of 5, `cuda.synchronize` around each call).

## Pipeline as implemented

* **Stage 1a** — per-32×32-block mean of *pixelwise* |frame_t − frame_{t−1}| (not the
  difference of block means, which cancels motion inside a block), adaptive threshold
  `thr = max(TAU_ABS, 2·q10)`.
* **Stage 1b** — pre-dilation 3×3 majority filter, dirty iff ≥5/9 neighbours dirty.
* **Stage 2** — morphological halo dilation, one block radius.
* **Stage 3** — projection onto the vision token grid (32 px block = vision patch 16 ×
  temporal patch 2 × spatial merge 2); a temporal patch is recomputed if **any** of its
  two frames is dirty (`any`, not `all` — the `all` variant silently drops the first
  temporal patch of every clip).
* **Stage 4** — full flush when `ŝ > 0.50` **or** when the frame's mean block difference
  exceeds `GAMMA × NOISE_MEAN` (global camera pan). Flush means every token is computed.
* **Stage 5** — static tokens are dropped from the multimodal prefill and their features
  are *reused*: retained tokens carry the **same** feature tensor, so no interpolation
  error is introduced (`feature_reuse_rel_l2 = 0.0` exactly, `results/phase4_video.json`).

## Results (dense vs P2, same weights, same kernel path)

| clip | tokens | reduction | ŝ (frames 1..7) | flush | prefill ms | speedup | logit KL | top-1 | scene-cut max\|Δ\| | VQA dense→P2 |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 static surveillance | 247→100 | 59.5% | 0 for all | 1 (frame 0) | 25.19→13.71 | **1.837×** | 1e-6 | 1.00 | 0.0 | 1.00→1.00 |
| V2 local motion | 247→110 | 55.5% | 0,0,0,0.04,0,0.02 | 1 | 23.00→13.13 | **1.751×** | 1e-6 | 1.00 | 0.0 | 1.00→1.00 |
| V3 fast natural object | 247→100 | 59.5% | 0 for all | 1 | 23.09→12.98 | **1.778×** | 1e-6 | 1.00 | 0.0 | 1.00→1.00 |
| V4 camera pan | 247→247 | 0% | 0 but mean diff 0.030 > 0.0255 | 8 | 23.84→23.85 | 0.999× | 0.0 | 1.00 | 0.0 | 1.00→1.00 |
| V5 illumination change | 247→100 | 59.5% | 0 for all | 1 | 23.20→12.88 | **1.801×** | 1e-6 | 1.00 | 0.0 | 1.00→1.00 |
| V6 multi-shot cuts | 247→247 | 0% | 0.551–0.592 > 0.50 | 8 | 23.96→24.02 | 0.998× | 0.0 | 1.00 | 0.0 | 1.00→1.00 |

* Prefill latency is 23–25 ms dense (247 tokens) and 13–14 ms skipped; the residual
  floor is the vision tower (9.5 ms, unaffected because the tower still encodes every
  frame — see limitations).
* **Scene-cut safety is exact**: with the flush fallback active, the prefill output is
  bit-identical to dense (`max|Δ| = 0.0`, KL `0.0`) on both the pan clip and the cut
  clip, satisfying the 0.000000 requirement.
* **Quality is unchanged** on 24 position-balanced MC questions (2 distractors × 2
  orders × 6 clips): 1.00 dense, 1.00 P2. The likelihood margins are also preserved
  (e.g. V2 `−5.11/−6.95` dense vs `−5.19/−7.35` skipped).
* **The global-motion gate is what saves V4.** With the ŝ>0.5 rule alone the pan clip
  produced ŝ = 0 and would have been skipped frame-by-frame (the pre-dilation majority
  filter erases the uniformly-scattered pan signal) — a silent failure mode that the
  gate removes at the cost of 0% savings on pans, which is the correct trade.

## Joint accounting with Phase 2/3 block sparsity

P2 removes tokens; π block sparsity removes weight bytes per token. They multiply in
work: token ratio 0.405 (V1) × byte ratio 0.79 (Variant A at τ=0.5, `3d²+0.5d²/3.56`
over `4d²`) = **0.32× work, a 3.1× ceiling**. Measured components: 1.84× (GPU prefill,
token skipping) and 1.17× (CPU kernel, block skipping) → 2.15× combined where both can
be realised. The honest caveat: on GPU we only have the masked-dense gating path, which
does not skip weight reads, so a fused Triton block-sparse kernel is required to collect
the block-sparsity factor on the GPU — that is the main piece of future work.

## Limitations recorded

1. Static tokens are *dropped*, not recomputed cheaply, so attention context shrinks;
   the effect is measured (KL ≤ 1e-6, top-1 1.0, VQA 1.00) but on an easy 6-clip task.
2. The vision tower still processes every frame (8.56% of checkpoint bytes); only the
   language prefill is shortened. Encoding only dirty blocks inside the tower requires a
   patch-sparse vision kernel.
3. Thresholds were calibrated on one static clip; the global-motion gate needs a
   per-camera noise floor in deployment (documented, not automated).
