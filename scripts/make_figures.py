"""Generate the report figures from the measured JSONs (no synthetic data)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "/home/user/Downloads/CM-LFM3/results", "/home/user/Downloads/CM-LFM3/figs"
import os; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.3, "lines.markersize": 4})

ab = json.load(open(f"{R}/phase2_ablation.json"))
dense = ab["dense"]
STY = {"native_topk": ("native top-K", "o", "#1f77b4"), "perm_topk": ("permuted top-K", "s", "#aec7e8"),
       "cats": ("CATS block", "^", "#2ca02c"), "teal": ("TEAL block", "v", "#98df8a"),
       "ours": ("ours (pi+theta+alpha)", "D", "#d62728"), "ours_pt": ("ours_pt (per-token)", "P", "#ff7f0e"),
       "wanda": ("Wanda 2:4", "*", "#7f7f7f")}

for mode, fname in (("ffn", "fig_ffn_ppl"), ("conv", "fig_conv_ppl"), ("both", "fig_both_ppl")):
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 2.9))
    for m in ab["methods"]:
        rows = sorted([r for r in ab["rows"] if r["mode"] == mode and r["method"] == m],
                      key=lambda r: r["tau"])
        if not rows:
            continue
        xs = [r["tau"] for r in rows]
        ys = [r["ppl_p3"] - dense["ppl_p3"] for r in rows]
        lab, mk, c = STY[m]
        ax.plot(xs, ys, marker=mk, color=c, label=lab)
    ax.set_xlabel(r"budget $\tau$")
    ax.set_ylabel(r"$\Delta$PPL (chat-conditioned)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title({"ffn": "FFN gating only", "conv": "LFM3Conv gating only",
                  "both": "both operators gated"}[mode], fontsize=9)
    if mode == "ffn":
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(f"{F}/{fname}.pdf"); fig.savefig(f"{F}/{fname}.png"); plt.close(fig)

# energy retention (revised): retained fraction from the audit, against the 1 - tau bound
au = json.load(open(f"{R}/phase2_energy_audit.json"))
rows = au["rows"]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.7), sharey=True)
for ax, fam, title in ((axes[0], "ffn", "text FFN"), (axes[1], "conv", "LFM3Conv")):
    rr = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["tau"])
    taus = [r["tau"] for r in rr]
    for order, ls, mk in (("native", "-", "o"), ("gram", "--", "s"), ("coact", ":", "^")):
        ax.plot(taus, [r[f"{order}_thr"] for r in rr], ls=ls, marker=mk, label=order)
    ax.plot(taus, [1.0 - t for t in taus], "k:", lw=1.0, label=r"$1-\tau$ bound")
    ax.set_title(title, fontsize=9); ax.set_xlabel(r"$\tau$")
    ax.set_ylim(0.75, 1.005)
axes[0].set_ylabel("retained energy fraction")
axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{F}/fig_energy.pdf"); fig.savefig(f"{F}/fig_energy.png"); plt.close(fig)

# video: tokens and speedup
v = json.load(open(f"{R}/phase4_video.json"))["clips"]
names = list(v)
fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
axes[0].bar(range(len(names)), [v[k]["tokens_dense"] for k in names], width=.38, label="dense")
axes[0].bar([i + .38 for i in range(len(names))], [v[k]["tokens_skipped"] for k in names],
            width=.38, label="P2")
axes[0].set_xticks([i + .19 for i in range(len(names))])
axes[0].set_xticklabels([k.split("_")[0] for k in names], fontsize=7)
axes[0].set_ylabel("multimodal prefill tokens"); axes[0].legend(fontsize=7, frameon=False)
axes[1].bar(range(len(names)), [v[k]["prefill_speedup"] for k in names], width=.6, color="#2ca02c")
axes[1].axhline(1.0, color="k", lw=.8, ls="--")
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels([k.split("_")[0] for k in names], fontsize=7)
axes[1].set_ylabel("prefill speedup"); axes[1].set_ylim(0.9, 2.0)
fig.tight_layout(); fig.savefig(f"{F}/fig_video.pdf"); fig.savefig(f"{F}/fig_video.png"); plt.close(fig)

# kernel scaling
k = json.load(open(f"{R}/phase3_kernels.json"))
print("phase3 keys", list(k)[:8])
print("saved figures to", F)
