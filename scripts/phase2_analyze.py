#!/usr/bin/env python3
"""Phase 2 analysis: P2/P3 tables, area-under-the-curve ranking, Stream Square Law."""
import json
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
METHODS = ["native_topk", "perm_topk", "cats", "teal", "ours", "ours_pt", "wanda"]
SHORT = {"native_topk": "Native-K", "perm_topk": "Perm-K", "cats": "CATS", "teal": "TEAL",
         "ours": "Ours", "ours_pt": "Ours-PT", "wanda": "Wanda2:4"}
MODES = ["ffn", "conv", "both"]


def load():
    with open(os.path.join(RES, "phase2_ablation.json")) as f:
        return json.load(f)


def r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / ss, float(coef[0]), float(coef[1])


def main():
    d = load()
    dense, rows = d["dense"], d["rows"]
    out = ["# Phase 2 ablation", "",
           f"dense: P2 PPL **{dense['ppl_p2']:.4f}** (raw WikiText, continuous KV, "
           f"{dense['tokens_p2']} tokens, span {dense['span_p2']}); "
           f"P3 PPL **{dense['ppl_p3']:.4f}** (chat-conditioned assistant span, "
           f"{dense['tokens_p3']} tokens, span {dense['span_p3']}).", ""]

    for metric, label in (("ppl_p3", "P3 chat-conditioned PPL (in-distribution)"),
                          ("ppl_p2", "P2 raw-WikiText PPL (off-distribution)")):
        out += [f"## {label}", ""]
        for mode in MODES:
            out += [f"### mode = {mode}", "",
                    "| tau | " + " | ".join(SHORT[m] for m in METHODS) + " |",
                    "|---" * (len(METHODS) + 1) + "|"]
            for tau in (0.10, 0.20, 0.30, 0.40, 0.50):
                cells = []
                for mth in METHODS:
                    r = [r for r in rows if r["mode"] == mode and r["method"] == mth
                         and abs(r["tau"] - tau) < 1e-6]
                    cells.append(f"{r[0][metric]:.1f}" if r else "-")
                out.append(f"| {tau:.2f} | " + " | ".join(cells) + " |")
            out.append("")

    # AUC ranking: mean delta PPL over the five budgets, per mode
    out += ["## Mean delta over budgets (lower is better)", "",
            "| mode | " + " | ".join(SHORT[m] for m in METHODS) + " |",
            "|---" * (len(METHODS) + 1) + "|"]
    for mode in MODES:
        cells = []
        for mth in METHODS:
            ds3 = [r["ppl_p3"] - dense["ppl_p3"] for r in rows
                   if r["mode"] == mode and r["method"] == mth]
            cells.append(f"{np.mean(ds3):+.2f}" if ds3 else "-")
        out.append(f"| {mode} | " + " | ".join(cells) + " |")
    out.append("")

    # Stream Square Law: delta PPL vs squared relative stream error
    out += ["## Stream Square Law: dPPL = a * rel2^2 + b", "",
            "| subset | metric | a | b | R^2 | n |", "|---|---|---|---|---|---|"]
    subsets = [("all", rows)]
    for mode in MODES:
        subsets.append((f"mode={mode}", [r for r in rows if r["mode"] == mode]))
    for mth in METHODS:
        subsets.append((f"method={SHORT[mth]}", [r for r in rows if r["method"] == mth]))
    law = {}
    for name, sub in subsets:
        for metric in ("ppl_p3", "ppl_p2"):
            xs = [r["rel2"] ** 2 for r in sub]
            ys = [r[metric] / (dense[metric]) for r in sub]  # relative delta
            if len(sub) < 3:
                continue
            v, a, b = r2(xs, ys)
            law[f"{name}|{metric}"] = {"slope": a, "intercept": b, "r2": v, "n": len(sub)}
            out.append(f"| {name} | {metric} | {a:.4f} | {b:+.4f} | {v:.4f} | {len(sub)} |")
    out.append("")

    with open(os.path.join(RES, "phase2_analysis.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(RES, "phase2_law.json"), "w") as f:
        json.dump(law, f, indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
