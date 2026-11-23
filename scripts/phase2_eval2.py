#!/usr/bin/env python3
"""Phase 2 ablation, second harness: P2 (raw continuous) + P3 (chat) + probe metrics.

Why a second harness: the first one showed that on *raw* WikiText -- which is far
off-distribution for this chat/VL model (dense PPL 62.8 vs 42.7 in-distribution)
-- masking the low-energy half of the FFN can *lower* PPL.  That is a real
measurement, not a bug (single-forward check: dense 130.11, CATS 111.50, native
top-K 154.99, random blocks 1350.9), but it means a PPL gain on raw text cannot be
read as "pruning helped".  We therefore report every configuration under three
protocols:

  P2  raw WikiText, continuous KV, 8 x 1024 tokens (tokens [32768, 40960))
  P3  chat-conditioned, assistant span only, 24 x 256 tokens ([40960, 47104))
  probe 128 tokens ([32640, 32768)) -> logit KL / RMSE / Top-1 / stream rel-L2

Wanda is included here because it mutates weights (restored afterwards).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import lfm3_sparse as S  # noqa: E402

M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CAL = os.path.join(RES, "phase2_calibration.pt")
METHODS = ["native_topk", "perm_topk", "cats", "teal", "ours", "ours_pt", "wanda"]
TAUS = [0.10, 0.20, 0.30, 0.40, 0.50]
MODES = ["ffn", "conv", "both"]


def registry(model):
    lm = model.model.language_model
    ffn = [(f"L{i:02d}.ffn", l.feed_forward) for i, l in enumerate(lm.layers)]
    conv = [(f"L{i:02d}.conv", l.conv) for i, l in enumerate(lm.layers) if hasattr(l, "conv")]
    return ffn, conv


def mods_for(mode, ffn, conv):
    if mode == "ffn":
        return [(n, m, "ffn") for n, m in ffn]
    if mode == "conv":
        return [(n, m, "conv") for n, m in conv]
    return ([(n, m, "ffn") for n, m in ffn] + [(n, m, "conv") for n, m in conv])


def cfg_for(method, tau, cal, mods):
    S.CFG.on = True
    S.CFG.alpha = True
    S.CFG.pt = method == "ours_pt"
    S.CFG.method = "ours" if method == "ours_pt" else method
    S.CFG.tau = tau
    if method in ("perm_topk", "ours", "ours_pt"):
        key = "gram"
    elif method == "cats":
        key = "coact"
    else:
        key = "native"
    for nm, m, kind in mods:
        if key == "native":
            d = 7680 if kind == "ffn" else 2048
            pi = torch.arange(d, device="cuda")
        else:
            pi = cal[nm][f"pi_{key}"].cuda()
        th = {t: float(cal[nm][f"theta_{key}"][t]) for t in TAUS}
        S.ST[id(m)] = {"pi": pi, "inv": pi.argsort(), "theta": th}


@torch.inference_mode()
def p2_ppl(model, chunks):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    tot, n, prev = 0.0, 0, None
    for ch in chunks:
        lg = model(input_ids=ch, past_key_values=cache, use_cache=True).logits.float()[0]
        rows = lg[:-1] if prev is None else torch.cat([prev.unsqueeze(0), lg[:-1]], 0)
        tgt = ch[0, 1:] if prev is None else ch[0]
        tot += F.cross_entropy(rows, tgt, reduction="sum").item()
        n += tgt.numel()
        prev = lg[-1].clone()
        del lg, rows
        gc.collect()
        torch.cuda.empty_cache()
    return float(np.exp(tot / n)), n


def build_p3(tok, ids):
    wins = []
    for i in range(24):
        seg = ids[40960 + i * 256:40960 + (i + 1) * 256]
        prefix, suffix = tok.decode(seg[:32]), tok.decode(seg[32:])
        msgs = [{"role": "user", "content": [{"type": "text", "text":
                 "Continue the following text exactly as it appears, without commentary.\n\n" + prefix}]},
                {"role": "assistant", "content": [{"type": "text", "text": suffix}]}]
        full = tok.apply_chat_template(msgs, tokenize=False)
        head, _, tail = full.rpartition("assistant\n")
        lead = head + "assistant\n"
        ids_full = tok(lead + tail, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        n0 = tok(lead, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
        wins.append((ids_full, n0))
    return wins


@torch.inference_mode()
def p3_ppl(model, wins):
    tot, n = 0.0, 0
    for ids_full, n0 in wins:
        lg = model(input_ids=ids_full, use_cache=False).logits.float()
        lp = torch.log_softmax(lg[0, n0 - 1:-1], -1)
        tgt = ids_full[0, n0:]
        tot += -lp.gather(-1, tgt[:, None]).sum().item()
        n += tgt.numel()
        del lg, lp
    return float(np.exp(tot / n)), n


@torch.inference_mode()
def probe_logits(model, x):
    return model(input_ids=x, use_cache=False).logits.float()[0].cpu()


def metrics(lg, lg0):
    kl = F.kl_div(F.log_softmax(lg, -1), F.log_softmax(lg0, -1), log_target=True,
                  reduction="batchmean").item()
    rmse = (lg - lg0).pow(2).mean().sqrt().item()
    top1 = (lg.argmax(-1) == lg0.argmax(-1)).float().mean().item()
    rel = ((lg - lg0).norm() / lg0.norm()).item()
    return kl, rmse, top1, rel


def wanda_prepare(cal, mods, ffn_all, tau):
    """2:4 in groups of 4 along the input dim of w2 / out_proj, importance |w|*sqrt(E)."""
    saved = []
    for nm, m, kind in mods:
        lin = m.w2 if kind == "ffn" else m.out_proj
        if not hasattr(lin, "weight"):
            continue
        w = lin.weight.data
        imp = (w.float().abs() * torch.sqrt(
            cal[nm]["E_native"].mean(0).repeat_interleave(S.DB)).cuda()[None, :])
        g = imp.reshape(imp.shape[0], -1, 4)
        keep = g.topk(2, dim=-1).indices
        mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, keep, True).reshape(w.shape)
        saved.append((lin, lin.weight.data))
        lin.weight.data = (w.float() * mask).to(w.dtype)
    return saved


def wanda_restore(saved):
    for lin, w in saved:
        lin.weight.data = w


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    ids = tok(open(os.path.join(RES, "wikitext2_test.txt")).read()[:400000],
              return_tensors="pt", add_special_tokens=False).input_ids[0]
    chunks = [ids[32768 + i * 1024:32768 + (i + 1) * 1024].unsqueeze(0).cuda() for i in range(8)]
    xprobe = ids[32640:32768].unsqueeze(0).cuda()
    cal = torch.load(CAL, weights_only=False)
    model = AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    ffn, conv = registry(model)
    S.patch(type(ffn[0][1]), type(conv[0][1]))
    wins = build_p3(tok, ids)

    S.CFG.on = False
    S.ST.clear()
    lg0 = probe_logits(model, xprobe)
    t0 = time.time()
    d2, n2 = p2_ppl(model, chunks)
    d3, n3 = p3_ppl(model, wins)
    print(f"dense: P2 {d2:.4f} (n={n2})  P3 {d3:.4f} (n={n3})  [{time.time()-t0:.0f}s]", flush=True)
    rows = [{"mode": "dense", "method": "dense", "tau": 0.0, "ppl_p2": d2, "ppl_p3": d3,
             "kl": 0.0, "rmse": 0.0, "top1": 1.0, "rel2": 0.0}]

    for mode in MODES:
        mods = mods_for(mode, ffn, conv)
        for method in METHODS:
            if method == "wanda":
                # 2:4 is a fixed 50% pattern, independent of tau: measure once
                t0 = time.time()
                S.CFG.on = False
                saved = wanda_prepare(cal, mods, ffn, 0.5)
                lg = probe_logits(model, xprobe)
                kl, rmse, top1, rel = metrics(lg, lg0)
                p2, _ = p2_ppl(model, chunks)
                p3, _ = p3_ppl(model, wins)
                for tau in TAUS:
                    rows.append({"mode": mode, "method": "wanda", "tau": tau, "ppl_p2": p2,
                                 "ppl_p3": p3, "kl": kl, "rmse": rmse, "top1": top1,
                                 "rel2": rel, "note": "wanda 2:4 fixed density"})
                print(f"  {mode}|wanda|2:4  P2 {p2:9.4f} ({p2-d2:+8.4f})  P3 {p3:8.4f} "
                      f"({p3-d3:+7.4f})  KL {kl:.4f}  rmse {rmse:.4f}  top1 {top1:.4f}  "
                      f"rel2 {rel:.4f}  [{time.time()-t0:.0f}s]", flush=True)
                wanda_restore(saved)
                del saved
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for tau in TAUS:
                t0 = time.time()
                cfg_for(method, tau, cal, mods)
                lg = probe_logits(model, xprobe)
                kl, rmse, top1, rel = metrics(lg, lg0)
                p2, _ = p2_ppl(model, chunks)
                p3, _ = p3_ppl(model, wins)
                rows.append({"mode": mode, "method": method, "tau": tau, "ppl_p2": p2,
                             "ppl_p3": p3, "kl": kl, "rmse": rmse, "top1": top1, "rel2": rel})
                print(f"  {mode}|{method}|{tau:.2f}  P2 {p2:9.4f} ({p2-d2:+8.4f})  "
                      f"P3 {p3:8.4f} ({p3-d3:+7.4f})  KL {kl:.4f}  rmse {rmse:.4f}  "
                      f"top1 {top1:.4f}  rel2 {rel:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    out = {"dense": {"ppl_p2": d2, "ppl_p3": d3, "tokens_p2": n2, "tokens_p3": n3,
                     "span_p2": [32768, 40960], "span_p3": [40960, 47104]},
           "methods": METHODS, "rows": rows, "date": time.strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(RES, "phase2_ablation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/phase2_ablation.json")


if __name__ == "__main__":
    main()
