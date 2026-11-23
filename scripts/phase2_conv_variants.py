#!/usr/bin/env python3
"""LFM3Conv two-level skip study: Variant A (state-exact, gate y before out_proj)
vs Variant B (gate-first: zero B/C/X of low-gate-energy blocks, so the in_proj
slice W_x AND the out_proj block both become skippable)."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase2_eval2 as E

def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E.M, trust_remote_code=True)
    ids = tok(open(os.path.join(E.RES, "wikitext2_test.txt")).read()[:400000],
              return_tensors="pt", add_special_tokens=False).input_ids[0]
    chunks = [ids[32768+i*1024:32768+(i+1)*1024].unsqueeze(0).cuda() for i in range(8)]
    xprobe = ids[32640:32768].unsqueeze(0).cuda()
    cal = torch.load(E.CAL, weights_only=False)
    model = AutoModelForImageTextToText.from_pretrained(
        E.M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"":0},
        trust_remote_code=True).eval()
    ffn, conv = E.registry(model)
    E.S.patch(type(ffn[0][1]), type(conv[0][1]))
    wins = E.build_p3(tok, ids)
    E.S.CFG.on = False; E.S.ST.clear()
    lg0 = E.probe_logits(model, xprobe)
    d2, _ = E.p2_ppl(model, chunks); d3, _ = E.p3_ppl(model, wins)
    print(f"dense P2 {d2:.4f}  P3 {d3:.4f}", flush=True)
    mods = E.mods_for("conv", ffn, conv)
    rows = []
    for method in ("native_topk", "cats"):
        for var in ("A", "B"):
            for tau in (0.20, 0.30, 0.40):
                E.cfg_for(method, tau, cal, mods)
                E.S.CFG.variant = var
                kl, rmse, top1, rel = E.metrics(E.probe_logits(model, xprobe), lg0)
                p2, _ = E.p2_ppl(model, chunks); p3, _ = E.p3_ppl(model, wins)
                rows.append({"method": method, "variant": var, "tau": tau, "ppl_p2": p2,
                             "ppl_p3": p3, "kl": kl, "rmse": rmse, "top1": top1, "rel2": rel})
                print(f"  {method:11s} V-{var} tau={tau:.2f}  P2 {p2:8.4f} ({p2-d2:+7.4f})  "
                      f"P3 {p3:8.4f} ({p3-d3:+7.4f})  KL {kl:.4f}  top1 {top1:.4f}", flush=True)
    E.S.CFG.variant = "A"
    with open(os.path.join(E.RES, "phase2_conv_variants.json"), "w") as f:
        json.dump({"dense": {"ppl_p2": d2, "ppl_p3": d3}, "rows": rows}, f, indent=1)
    print("saved results/phase2_conv_variants.json")

if __name__ == "__main__":
    main()
