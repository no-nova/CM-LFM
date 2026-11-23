#!/usr/bin/env python3
"""Attribution run: does the norm compensation alpha help, and does a layer-global
threshold lose to a per-token one?  Mode "both" (FFN + LFM3Conv gated), 5 budgets,
methods ours (global theta) and ours_pt (per-token theta), each with alpha on/off.
"""
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase2_eval2 as E  # noqa: E402

TAUS = E.TAUS


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.M, trust_remote_code=True)
    ids = tok(open(os.path.join(E.RES, "wikitext2_test.txt")).read()[:400000],
              return_tensors="pt", add_special_tokens=False).input_ids[0]
    chunks = [ids[32768 + i * 1024:32768 + (i + 1) * 1024].unsqueeze(0).cuda() for i in range(8)]
    xprobe = ids[32640:32768].unsqueeze(0).cuda()
    cal = torch.load(E.CAL, weights_only=False)
    model = AutoModelForImageTextToText.from_pretrained(
        E.M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    ffn, conv = E.registry(model)
    E.S.patch(type(ffn[0][1]), type(conv[0][1]))
    wins = E.build_p3(tok, ids)
    E.S.CFG.on = False
    E.S.ST.clear()
    lg0 = E.probe_logits(model, xprobe)
    d2, _ = E.p2_ppl(model, chunks)
    d3, _ = E.p3_ppl(model, wins)
    print(f"dense P2 {d2:.4f}  P3 {d3:.4f}", flush=True)
    mods = E.mods_for("both", ffn, conv)
    rows = []
    for method in ("ours", "ours_pt"):
        for alpha in (True, False):
            for tau in TAUS:
                E.cfg_for(method, tau, cal, mods)
                E.S.CFG.alpha = alpha
                kl, rmse, top1, rel = E.metrics(E.probe_logits(model, xprobe), lg0)
                p2, _ = E.p2_ppl(model, chunks)
                p3, _ = E.p3_ppl(model, wins)
                rows.append({"method": method, "alpha": alpha, "tau": tau,
                             "ppl_p2": p2, "ppl_p3": p3, "kl": kl, "rmse": rmse,
                             "top1": top1, "rel2": rel})
                print(f"  {method:8s} alpha={int(alpha)} tau={tau:.2f}  P2 {p2:8.4f} "
                      f"({p2-d2:+7.4f})  P3 {p3:8.4f} ({p3-d3:+7.4f})  KL {kl:.4f} "
                      f"top1 {top1:.4f}", flush=True)
    out = {"dense": {"ppl_p2": d2, "ppl_p3": d3}, "rows": rows}
    with open(os.path.join(E.RES, "phase2_alpha.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/phase2_alpha.json")


if __name__ == "__main__":
    main()
