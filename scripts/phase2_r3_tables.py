#!/usr/bin/env python3
"""R3 ablation tables and figures from results/phase2_r3.json.

Emits results/tab_r3_{ffn,conv,both}.tex and figs/fig_r3_{ffn,conv}.pdf.
Columns are Delta-P3 (chat-conditioned, dense 36.0438) across the five block
budgets, then the mean Delta-P3 and the mean Delta-P2 (raw WikiText, dense
62.6275).  Best cell per budget is bold.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/user/Downloads/CM-LFM3"
TAUS = ["0.10", "0.20", "0.30", "0.40", "0.50"]
LABEL = {
    "native_topk": "native top-$K$",
    "native_wn": "native + $\\wn$ (weighted)",
    "cats": "CATS-block",
    "ours_rms": "ours \\rms ($\\pi_{\\mathrm{coact}}$, per-token RMS)",
    "ours_wn": "ours \\wn ($\\pi_{\\wn}$, output-norm weighted)",
    "wanda": "Wanda 2:4 (fixed ratio)",
}


def main() -> int:
    d = json.load(open(f"{ROOT}/results/phase2_r3.json"))
    runs, dense = d["runs"], d["dense"]
    for mode in ("ffn", "conv", "both"):
        methods = ["native_topk", "native_wn", "cats", "ours_rms", "ours_wn"]
        rows = {m: [runs[f"{mode}|{m}|{t}"] for t in TAUS] for m in methods}
        # Wanda 2:4 is a fixed ratio: one measurement, shown as a flat reference row
        wkey = f"{mode}|wanda|0.50"
        if wkey in runs:
            methods.append("wanda")
            rows["wanda"] = [runs[wkey]] * len(TAUS)
        # bold the best (lowest) dp3 per budget
        best = {i: min(range(len(methods)), key=lambda k: rows[methods[k]][i]["dp3"])
                for i in range(len(TAUS))}
        lines = [r"\begin{tabular}{l" + "r" * 7 + "}", r"\toprule",
                 "method & " + " & ".join(f"$\\tau{{=}}{t}$" for t in TAUS) +
                 r" & mean & mean $\Delta$P2 \\", r"\midrule"]
        for k, m in enumerate(methods):
            cells = []
            for i, t in enumerate(TAUS):
                v = rows[m][i]["dp3"]
                s = f"${v:+.2f}$"
                if best[i] == k:
                    s = f"$\\mathbf{{{v:+.2f}}}$"
                cells.append(s)
            d3 = np.mean([r["dp3"] for r in rows[m]])
            d2 = np.mean([r["dppl"] for r in rows[m]])
            lines.append(f"{LABEL[m]} & " + " & ".join(cells) +
                         f" & ${d3:+.2f}$ & ${d2:+.2f}$ \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
        open(f"{ROOT}/results/tab_r3_{mode}.tex", "w").write("\n".join(lines))
        print("\n".join(lines))

        if mode == "both":
            continue
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        xs = [float(t) for t in TAUS]
        for m in methods:
            ys = [max(r["dp3"], 1e-2) for r in rows[m]]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.3, label=LABEL[m].replace("$", ""))
        ax.set_yscale("log")
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        ax.set_xlabel(r"block budget $\tau$")
        ax.set_ylabel(r"$\Delta$PPL (chat-conditioned)")
        ax.set_title(("SwiGLU FFN" if mode == "ffn" else r"LFM3Conv") + " gating only")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=5.5, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{ROOT}/figs/fig_r3_{mode}.pdf")
        fig.savefig(f"{ROOT}/figs/fig_r3_{mode}.png", dpi=110)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
