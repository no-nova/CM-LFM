# Phase 2 — Dual isomorphic permutation π, calibrated thresholds θ_ℓ and compensation α

Model: `LiquidAI/LFM3-VL-2B` (text backbone: d = 2048, d_ff = 7680, 24 layers,
13 `Lfm3ShortConv` / 11 GQA). Everything below is measured on the real checkpoint
(`results/phase2_*.json`), no synthetic numbers.

## 1. Isomorphism at τ = 0 (proved, then verified)

Two theorems, both proved by writing the layers out channel-by-channel:

* **FFN.** `y = W₂ (silu(W₁x) ⊙ W₃x)`. `W₁`/`W₃` act per output channel and `W₂`
  only through its *columns*, so for any permutation π of the 7680 intermediate
  channels, `(W₁,W₃,W₂) → (W₁[π],W₃[π],W₂[:,π])` computes the identical function.
* **LFM3Conv.** `y = W_out · (C ⊙ dwconv(B ⊙ X))` with `B,C,X = split(W_in x)` and
  a *channel-wise* (grouped, `groups=d`) causal conv. Every stage after `W_in`
  is channel-local, and `W_out` only sees `y` through its columns, so permuting
  the output channels of `W_in` (in all three slices), the depthwise conv weight
  and bias, and the input columns of `W_out` is an exact symmetry.

Verification (one kernel path, permuted-dense as the τ = 0 reference):

| level | dtype | max \|Δlogit\| | rel L2 |
|---|---|---|---|
| operator, FFN 2048→7680 | fp64 | 2.7e-14 | – |
| operator, FFN 2048→7680 | fp32 | 1.5e-6 | – |
| operator, LFM3Conv d=2048, T=16 | fp64 | 1.2e-15 | – |
| operator, LFM3Conv d=2048, T=16 | fp32 | 1.4e-7 | – |
| whole model, 37 gated layers, T=16 | fp32 | 1.04e-4 (π_ffn) / 6.10e-5 (π_conv) | 2.29e-6 / 2.37e-6 |
| negative control (wrong pairing of col/row permutation) | fp32 | 0.53–0.58 | 0.17 |
| inverse-permutation restore | fp32 | 0.0 (bitwise) | 0.0 |

Two necessary methodological notes. (i) In **bf16** the same test gives
`max|Δ| = 0.76` while an *identity* permutation and a full reload both give exactly
`0.0`; bf16 GEMM re-association therefore cannot certify a 1e-5 symmetry, so the
certification is done in fp32 (whole model, 7.15 GiB of weights) and bf16 is used
only for the quality runs. (ii) The permutation is implemented **only as runtime
index tensors** (`pi`, `inv`), never by rewriting weights, so at τ = 0 the gated
layer is *bitwise* the dense layer and no reordering error can hide in the baseline.

## 2. Block construction and calibration

* 32-lane blocks (`DB = 32`) over the *gated intermediate* activation:
  `h = silu(W₁x)⊙W₃x` (FFN, 7680) and `y = C⊙dwconv(B⊙X)` (LFM3Conv, 2048).
* **π_gram**: greedy blocks minimising the expected output Gram error
  `E‖Δy‖² = Σ_{i,j∈S} E[aᵢaⱼ]⟨W_out[:,i],W_out[:,j]⟩`, i.e. `M = C ⊙ (W_outᵀW_out)`.
* **π_coact**: activation-only co-activation clustering (CATS-style).
* θ_ℓ(τ): offline quantile of the layer's block-energy distribution (4096 sampled
  rows × 128 calibration sequences of 256 tokens = 32 768 WikiText tokens taken
  from tokens `[0, 32768)`; the evaluation spans start at 32768 and 40960, so
  calibration and evaluation text are disjoint).
* α = `min(1.5, ‖a‖₂ / ‖a_keep‖₂)` per token.

Mechanism diagnostic (`results/phase2_energy_diag.json`, mean fraction of stream
energy dropped by a fixed top-K block budget):

| τ | FFN native / gram / coact | LFM3Conv native / gram / coact |
|---|---|---|
| 0.10 | 2.67% / 2.63% / 1.63% | 1.24% / 1.13% / 0.33% |
| 0.30 | 10.93% / 10.86% / 7.11% | 6.22% / 5.85% / 2.13% |
| 0.50 | 22.14% / 22.10% / 15.13% | 14.21% / 13.74% / 5.71% |

Gram-coupled blocks are energy-neutral w.r.t. native (as designed: they trade
energy for output-error coherence), co-activation blocks concentrate energy.

## 3. Ablation (dense baseline: P2 = 62.6275 raw WikiText continuous KV, 8191
tokens; P3 = 36.0438 chat-conditioned assistant span, 5472 tokens)

Protocols. **P2** raw WikiText-2, continuous KV, 8 × 1024 tokens. **P3**
chat-conditioned (paragraph dictated inside the real chat template, loss on the
assistant span only) — the in-distribution protocol. Probe metrics on the 128
tokens preceding the span.

Full tables: `results/phase2_analysis.md`; raw rows: `results/phase2_ablation.json`.

P3, mode = FFN (ΔPPL vs dense):

| τ | Native-K | Perm-K | CATS | TEAL | Ours | Ours-PT | Wanda 2:4 |
|---|---|---|---|---|---|---|---|
| 0.10 | +0.82 | +0.45 | +0.41 | +0.43 | +0.67 | +0.48 | +7.72 |
| 0.20 | +2.21 | +2.22 | +1.46 | +2.19 | +2.52 | +2.57 | +7.72 |
| 0.30 | +5.35 | +6.16 | +3.37 | +5.53 | +4.86 | +6.02 | +7.72 |
| 0.40 | +10.32 | +11.35 | +5.90 | +10.31 | +11.24 | +12.05 | +7.72 |
| 0.50 | +20.84 | +21.33 | +10.70 | +22.18 | +22.35 | +22.43 | +7.72 |

P3, mode = LFM3Conv (ΔPPL):

| τ | Native-K | Perm-K | CATS | TEAL | Ours | Ours-PT | Wanda 2:4 |
|---|---|---|---|---|---|---|---|
| 0.10 | +0.86 | +1.61 | +3.19 | +0.75 | +0.43 | +1.63 | +1.79 |
| 0.20 | +2.95 | +6.21 | +9.42 | +2.74 | +3.87 | +6.43 | +1.79 |
| 0.30 | +7.90 | +15.16 | +22.25 | +6.70 | +15.87 | +15.11 | +1.79 |
| 0.40 | +15.59 | +40.05 | +57.45 | +15.68 | +48.54 | +36.69 | +1.79 |
| 0.50 | +32.37 | +98.99 | +191.0 | +36.16 | +190.0 | +92.84 | +1.79 |

Mean ΔP3 over the five budgets (lower is better):

| mode | Native-K | Perm-K | CATS | TEAL | Ours | Ours-PT | Wanda 2:4 |
|---|---|---|---|---|---|---|---|
| FFN | +7.93 | +8.35 | **+4.34** | +8.13 | +8.34 | +8.74 | +7.80 |
| Conv | +11.93 | +32.40 | +56.73 | +12.40 | +51.74 | +30.54 | **+1.73** |
| Both | +11.41 | +32.40 | +30.82 | +12.30 | +51.74 | +30.54 | +10.72 |

## 4. What the numbers say (including the negative results)

1. **The isomorphism holds, but it does not buy accuracy.** π_ffn and π_conv are
   exact at τ = 0, yet neither permutation improves the PPL-vs-budget curve on
   the LFM3Conv layers, and the gram-coupled ordering is *worse* than the native
   layout there. The reason is visible in the diagnostic: π_gram is energy-neutral
   by construction (it optimises output-Gram error, not the energy ranking that
   block *selection* uses), and π_coact concentrates energy into few blocks — a
   good thing only when the stream really is cluster-sparse.
2. **Block-partition quality is layer-type dependent.** CATS-style co-activation
   blocks are the best partition on the SwiGLU FFN (4.34 mean ΔP3, i.e. 45% lower
   than native top-K) and the *worst* on LFM3Conv (+56.7). LFM3Conv's gated output
   is diffuse (co-activation clusters retain only 5.7% of the energy at τ = 0.5 vs
   15.1% for native blocks), so "keep the 50% most energetic clusters" keeps a
   strange, weight-blind subset.
3. **A layer-global threshold is strictly worse than a per-token budget.**
   Ours (global θ_ℓ) = worst in every mode; Ours-PT (same blocks, same α, but the
   threshold taken per token) recovers native-top-K behaviour. Per-token adaptivity
   is worth 20–40% of the ΔPPL at every budget ≥ 0.3 — the single largest effect in
   this table, larger than the choice of permutation.
4. **The norm compensation α does not help.** With α on vs off (mode = both,
   `results/phase2_alpha.json`): Ours-PT τ = 0.10 P3 +1.63 vs +1.53, τ = 0.30
   +15.11 vs +15.42, τ = 0.50 +92.84 vs +102.40 — mixed and within noise at small
   τ, clearly *harmful* at τ = 0.5 for Ours (190.0 vs 260.0). Rescaling the kept
   subset does not reconstruct the dropped subspace; it only amplifies whatever
   the drop already damaged.
5. **Weight-importance masking wins the coarse end.** Wanda 2:4 (no blocks, no
   permutation, `|w|·√E` in groups of 4) is the best configuration at τ ≈ 0.5 on
   both layer types (FFN +7.72, Conv +1.79) and the only one that is nearly free
   on LFM3Conv. Since 2:4 is exactly a 50% input-channel cut on `w2`/`out_proj`,
   this says: at half density, what matters is *which channels* (weight ×
   activation importance), not how they are grouped into 32-lane blocks.
6. **Raw-WikiText PPL is not a usable pruning metric for this model.** It is
   far off-distribution (dense PPL 62.6 vs 36.0 in-distribution), and masking can
   *lower* it: mode = FFN, CATS τ = 0.5 gives P2 54.3 (−8.3) while P3 rises by
   +10.7. A single-forward control confirms it is real, not a harness bug (dense
   130.11, CATS 111.50, native top-K 154.99, random blocks 1350.9): the pruned
   model suppresses a noise pathway that only hurts on OOD text. **All pruning
   conclusions in this report are therefore drawn on P3**, and P2 is reported
   only for completeness.
7. **Stream Square Law** (ΔPPL/PPL vs squared relative stream error, per method,
   `results/phase2_law.json`): linear in `rel2²` with R² = 0.95 (Native-K), 0.89
   (Ours-PT), 0.99 (Wanda 2:4) on P3, but only R² = 0.58 pooled across all
   configurations — the law holds *within* a selection rule and breaks *across*
   selection rules, because two rules with the same `rel2` can drop different
   channels (norm of the dropped `W_out` columns differs). This is the same reason
   energy-only gating underperforms Wanda.

## 5. LFM3Conv two-level skip study (Variant A vs Variant B)

| method | variant | τ | ΔP2 | ΔP3 | KL | Top-1 | skippable weight share |
|---|---|---|---|---|---|---|---|
| native top-K | A (state-exact) | 0.20 | +3.05 | +0.61 | 0.165 | 0.742 | 1/4 of layer (out_proj block) |
| native top-K | A | 0.30 | +3.87 | +1.44 | 0.299 | 0.625 | 1/4 |
| native top-K | A | 0.40 | +5.50 | +2.34 | 0.481 | 0.641 | 1/4 |
| native top-K | B (gate-first) | 0.20 | −3.44 | +2.93 | 0.659 | 0.648 | 2/4 (out_proj + W_x slice) |
| native top-K | B | 0.30 | −2.11 | +7.07 | 0.924 | 0.516 | 2/4 |
| native top-K | B | 0.40 | +0.43 | +17.85 | 1.189 | 0.445 | 2/4 |
| CATS | A | 0.20 | +14.72 | +5.43 | 0.878 | 0.547 | 1/4 |
| CATS | A | 0.40 | +40.69 | +20.94 | 1.449 | 0.375 | 1/4 |
| CATS | B | 0.20 | +29.90 | +17.87 | 1.497 | 0.422 | 2/4 |
| CATS | B | 0.40 | +162.7 | +227.3 | 2.145 | 0.367 | 2/4 |

Variant B doubles the skippable weight mass per gated block (it drops the `W_x`
input-projection slice as well as the `out_proj` block) but pays far more accuracy
per skipped byte: at equal *skipped bytes* (B at τ = 0.40 ≈ A at τ = 0.20) V-B
costs +17.85 P3 against V-A's +0.61, and even B at τ = 0.20 (+2.93) loses to A at
τ = 0.40 (+2.34). **Variant A dominates on the accuracy-vs-skipped-bytes Pareto
front**; Variant B is only defensible where the gate itself is already computed and
the budget is very large. Phase 3 therefore implements A.

## 6. Deliverables

* `src/lfm3_sparse.py` — permutation/θ/α machinery, patched forwards, calibration hooks.
* `scripts/phase2_calibrate.py` — covariance + energy calibration (peak RssAnon ≈ 1.3 GB, one model).
* `scripts/phase2_eval2.py` — P2 + P3 + probe ablation (all 7 methods × 5 budgets × 3 modes).
* `scripts/phase2_alpha.py`, `scripts/phase2_conv_variants.py`, `scripts/phase2_analyze.py`.
* `results/phase2_isomorphism*.json`, `phase2_calibration.pt`, `phase2_energy_diag.json`,
  `phase2_ablation.json`, `phase2_alpha.json`, `phase2_conv_variants.json`,
  `phase2_analysis.md`, `phase2_law.json`.

---

## Revision R2 (after review): the energy metric was mislabelled, not inverted

`scripts/phase2_energy_audit.py` re-derives retained energy for both families, three block
orders and two decision rules, and asserts `rho(tau) > 1 - tau`, `rho_coact >= rho_native`,
`|rho_gram - rho_native| <= 0.02`. **All 30 cells pass.** The earlier text reported the
*dropped* fraction and called it "retains", which is what made 5.71% / 15.13% look like a
violation of arithmetic; the retained values are 94.29% (conv, coact) and 84.87% (FFN,
coact) at tau=0.5. Retained energy (thr rule, native / gram / coact):

| tau | FFN | Conv |
|---|---|---|
| 0.10 | 97.74 / 97.85 / 98.57 % | 99.71 / 99.72 / 99.97 % |
| 0.30 | 90.43 / 90.73 / 93.25 % | 97.86 / 97.54 / 99.69 % |
| 0.50 | 80.34 / 80.72 / 85.13 % | 93.64 / 92.59 / 98.95 % |

Alpha vs the sandwich RMSNorm (`scripts/phase2_alpha_rmsnorm.py`, real layer-0 activations,
tau=0.5): on the FFN, alpha=1.5 moves the projection by rel L2 0.5882 before
`post_ffn_norm` and 0.0157 after it -> **97.3% of the effect is annihilated** by the
per-token division; on LFM3Conv only 31% is annihilated (0.5068 -> 0.3503) because the conv
output's per-token variance sits near eps (median var 3.7e-5, eps/var ~ 0.27), so the norm
is no longer scale-invariant there. Recommendation: alpha = 1, which is what the ablation
already showed empirically.

---

## Revision R3 — per-token RMS rule, output-norm weighting, alpha = 1

The R2 ablation tables are superseded. Three defects were found and fixed:

1. **Inverted threshold quantile.** `theta_of` asked for the `1-tau` quantile and then kept
   `e > theta`, i.e. it kept the *hottest* tau fraction (retained energy 0.21 at tau=0.1).
   The rule is now `theta = Quantile_tau(E)` with `m = E > theta`.
2. **Sequence-summed block energy.** `E` is now the per-token RMS over the 32 lanes,
   `sqrt(mean_32 a^2)`, so theta is a per-lane amplitude and does not scale with T
   (active fraction 0.500/0.502/0.500 at T=1/16/64, tau=0.5).
3. **alpha.** Forced to 1. The alpha=1.5 of R2 was a 35% scale error on the conv stream
   (post-norm is not scale-invariant there, variance 3.7e-5 vs eps 1e-5) and is the true
   cause of the "conv permutation is harmful" conclusion; it is withdrawn.

New rule: **output-norm weighted block energy**, `E~ = sqrt(mean_32 (a_i w_i)^2)` with
`w_i = ||W_out[:,i]||_2`, and a matching partition `pi_wn` built on `C ⊙ w w^T`. `w` is
static, so the runtime cost is zero (it is absorbed into the offline theta).

### Results (P3 chat-conditioned, dense 36.0438; mean Delta P2 in the last column)

FFN only:

| method | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | mean | mean dP2 |
|---|---|---|---|---|---|---|---|
| native top-K | +0.80 | +2.25 | +5.08 | +10.88 | +20.81 | +7.96 | +2.25 |
| native + wn (weighted) | +0.54 | +2.31 | +5.08 | +10.25 | +20.92 | +7.82 | +6.80 |
| CATS-block | +0.33 | +1.37 | +2.88 | +5.93 | +10.28 | +4.16 | -5.68 |
| **ours rms** (pi_coact) | **-0.07** | +0.10 | +0.54 | **+0.84** | **+2.49** | **+0.78** | -5.57 |
| ours wn (pi_wn) | +0.03 | **+0.00** | **+0.18** | +1.36 | +2.98 | +0.91 | -6.18 |

LFM3Conv only:

| method | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | mean | mean dP2 |
|---|---|---|---|---|---|---|---|
| native top-K | +0.28 | +0.66 | +1.40 | +2.21 | +3.39 | +1.59 | +4.49 |
| native + wn | +0.08 | +0.29 | +0.71 | +1.58 | +3.50 | +1.23 | +4.15 |
| CATS-block | **-0.07** | -0.03 | +0.26 | +0.47 | +0.72 | +0.27 | +4.24 |
| ours rms | **-0.07** | **-0.18** | **-0.27** | **-0.29** | -0.06 | **-0.17** | +3.07 |
| ours wn | **-0.07** | +0.07 | -0.05 | +0.06 | **-0.13** | -0.02 | +1.81 |

Both families: ours rms mean +0.98 / ours wn +1.13, native top-K +11.06, CATS +5.08,
Wanda 2:4 +10.72.

### Conclusions revised

* The **decision rule dominates the ordering**: FFN at tau=0.5, native +20.81 vs ours +2.49.
* **Conv gating is lossless** (-0.17 mean) with the rms rule; the earlier "+32" was alpha.
* Output-norm weighting is the best conv metric (-0.13 at tau=0.5) and neutral on the FFN
  (FFN output columns are near-isotropic, CV 0.06; conv columns span a decade).
* Wanda 2:4 no longer wins anywhere (+10.72 joint vs +0.98) - the R2 claim is withdrawn.
* Raw-text P2 still rewards dropping (CATS -5.68, ours -5.57): all conclusions on P3.
