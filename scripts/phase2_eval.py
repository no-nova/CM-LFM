#!/usr/bin/env python3
"""Phase 2 ablation: pi_ffn / pi_conv block sparsity on LFM3-VL-2B (text backbone).

One process, one model.  Evaluation span = WikiText-2 tokens [32768, 40960)
(held out from the 32k calibration prefix).  Continuous KV PPL: the sequence is
streamed in 1024-token chunks with a persistent DynamicCache, so tokens 1..N
see the full prefix (no window restarts, unlike the Phase-1 sliding protocol).

Methods (all quality-measured masked-dense; skipping is Phase 3):
  native_topk : identity order, fixed top-K blocks                 (tau)
  perm_topk   : gram order,     fixed top-K blocks                 (tau)
  cats        : coact order,    fixed top-K blocks                 (tau)
  teal        : identity order, calibrated threshold theta_l       (tau)
  ours        : gram order,     theta_l + norm compensation alpha  (tau)
  wanda       : 2:4 semi-structured weight pruning of the gated projection
"""
import gc, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user/Downloads/CM-LFM3/src")
import lfm3_sparse as S                                    # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
RES = "/home/user/Downloads/CM-LFM3/results"
CAL = os.path.join(RES, "phase2_calibration.pt")
TAUS = [0.10, 0.20, 0.30, 0.40, 0.50]
PROBE = 128
N_CHUNK, CHUNK = 8, 1024
EVAL_START = 32768
D_FFN, D_CONV = 7680, 2048


# --------------------------------------------------------------------- helpers
def registry(model):
    lm = model.model.language_model
    ffn = [(f"L{i:02d}.ffn", l.feed_forward) for i, l in enumerate(lm.layers)]
    conv = [(f"L{i:02d}.conv", l.conv) for i, l in enumerate(lm.layers) if hasattr(l, "conv")]
    return ffn, conv


@torch.inference_mode()
def stream_nll(model, chunks, want_logits=False):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    tot, n, logits_out = 0.0, 0, []
    prev = None
    for ch in chunks:
        out = model(input_ids=ch, past_key_values=cache, use_cache=True)
        lg = out.logits.float()[0]
        T = lg.shape[0]
        if prev is None:
            rows, tgt = lg[:-1], ch[0, 1:]
        else:
            rows = torch.cat([prev.unsqueeze(0), lg[:-1]], 0)
            tgt = ch[0]
        tot += F.cross_entropy(rows, tgt, reduction="sum").item()
        n += tgt.numel()
        prev = lg[-1].clone()
        if want_logits:
            logits_out.append(lg.cpu())
        del out, lg, rows
        gc.collect(); torch.cuda.empty_cache()
    return (tot, n, logits_out)


@torch.inference_mode()
def probe_forward(model, ids, modules, want):
    """One forward on the probe; returns per-module gated activations if asked."""
    got = {}
    hooks = []
    if want:
        for nm, m in modules:
            def fn(mod, inp, out, nm=nm):
                got[nm] = out
            hooks.append(m.register_forward_hook(fn))
    logits = model(input_ids=ids).logits.float()[0]
    for h in hooks:
        h.remove()
    return logits.cpu(), {k: v.float().reshape(-1, v.shape[-1]).cpu() for k, v in got.items()}


def apply_config(cfgmethod, mode, tau, st, model_modules):
    """Set ST/CFG for a runtime-gating method."""
    S.CFG.on = True
    S.CFG.method = cfgmethod
    S.CFG.tau = tau
    S.CFG.alpha = True
    for nm, m, order in model_modules:
        kind = "gram" if cfgmethod in ("perm_topk", "ours") else ("coact" if cfgmethod == "cats" else "native")
        pi = S.identity(st[nm]["d"]) if kind == "native" else st[nm][f"pi_{kind}"].long().cuda()
        S.set_perm(m, pi)
        S.ST[id(m)]["theta"] = dict(st[nm][f"theta_{kind}"])


def wanda_prune(modules, cdiag, sparsity=0.5):
    """2:4 semi-structured pruning of the *input* dim of a Linear weight."""
    saved = []
    for nm, m in modules:
        for name in ("w2", "out_proj"):
            if not hasattr(m, name):
                continue
            lin = getattr(m, name)
            w = lin.weight.data                       # (out, in)
            metric = w.float().abs() * cdiag[id(lin)].sqrt().unsqueeze(0)
            g = metric.reshape(w.shape[0], w.shape[1] // 4, 4)
            k = int(round((1 - sparsity) * 4))
            idx = g.topk(k, dim=-1).indices
            mask = torch.zeros_like(g).scatter_(-1, idx, 1.0).reshape(w.shape)
            saved.append((lin, w))
            lin.weight.data = (w.float() * mask).to(w.dtype)
    return saved


def restore(saved):
    for lin, w in saved:
        lin.weight.data = w


# ------------------------------------------------------------------------ main
@torch.inference_mode()
def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    ids = tok(open(os.path.join(RES, "wikitext2_test.txt")).read()[:400000],
              return_tensors="pt", add_special_tokens=False).input_ids[0]
    seq = ids[EVAL_START:EVAL_START + N_CHUNK * CHUNK]
    chunks = [seq[i * CHUNK:(i + 1) * CHUNK].unsqueeze(0).cuda() for i in range(N_CHUNK)]
    probe = ids[EVAL_START - PROBE:EVAL_START].unsqueeze(0).cuda()

    model = AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    ffn, conv = registry(model)
    S.patch(type(ffn[0][1]), type(conv[0][1]))
    allmods = ffn + conv
    cal = torch.load(CAL, map_location="cpu", weights_only=False)
    st = {nm: {"d": (D_FFN if nm.endswith("ffn") else D_CONV), **cal[nm]} for nm, _ in allmods}
    name_of = {id(m): nm for nm, m in allmods}
    res = {"setup": {"eval_start": EVAL_START, "tokens": N_CHUNK * CHUNK, "probe": PROBE,
                     "methods": ["native_topk", "perm_topk", "cats", "teal", "ours", "ours_pt", "wanda"],
                     "taus": TAUS}, "runs": {}}

    # baseline: dense (patched forwards, CFG.on = False)
    S.CFG.on = False
    S.ST.clear()
    nll, n, _ = stream_nll(model, chunks)
    base_ppl = float(torch.exp(torch.tensor(nll / n)))
    lg0, act0 = probe_forward(model, probe, allmods, want=True)
    print(f"dense baseline: PPL {base_ppl:.4f} over {n} tokens  ({time.time()-t0:.0f}s)", flush=True)
    res["dense"] = {"ppl": base_ppl, "nll_sum": nll, "n": n}
    res["dense"]["probe_act_norm2"] = {k: float((v ** 2).sum()) for k, v in act0.items()}

    # per-config loop
    for mode in ("ffn", "conv", "both"):
        act_mods = {"ffn": ffn, "conv": conv, "both": allmods}[mode]
        mods3 = [(nm, m, None) for nm, m in act_mods]
        S.CFG.mode = mode
        S.ST.clear()
        S.CFG.on = True
        for method in ("native_topk", "perm_topk", "cats", "teal", "ours", "ours_pt"):
            for tau in TAUS:
                key = f"{mode}|{method}|{tau:.2f}"
                for nm, m, _ in mods3:
                    kind = "gram" if method in ("perm_topk", "ours", "ours_pt") else ("coact" if method == "cats" else "native")
                    pi = S.identity(st[nm]["d"]) if kind == "native" else st[nm][f"pi_{kind}"].long().cuda()
                    S.set_perm(m, pi)
                    S.ST[id(m)]["theta"] = dict(st[nm][f"theta_{kind}"])
                S.CFG.method = "ours" if method == "ours_pt" else method
                S.CFG.pt = method == "ours_pt"
                S.CFG.tau = tau
                nll, n, _ = stream_nll(model, chunks)
                ppl = float(torch.exp(torch.tensor(nll / n)))
                lg, act = probe_forward(model, probe, allmods, want=True)
                # logit metrics vs dense
                p, q = F.log_softmax(lg0, -1), F.log_softmax(lg, -1)
                kl = float(F.kl_div(p, q, log_target=True, reduction="batchmean").item())
                rmse = float((lg - lg0).pow(2).mean().sqrt())
                top1 = float((lg.argmax(-1) == lg0.argmax(-1)).float().mean())
                # stream relative L2 error over the gated modules of this mode
                errs = []
                for nm, _ in act_mods:
                    d = (act[nm] - act0[nm])
                    errs.append(float(d.pow(2).sum() / act0[nm].pow(2).sum().clamp_min(1e-12)))
                stream_rel2 = sum(errs) / len(errs)
                res["runs"][key] = {"ppl": ppl, "dppl": ppl - base_ppl, "kl": kl, "rmse": rmse,
                                    "top1": top1, "stream_rel2": stream_rel2,
                                    "dropped_energy_mean": None}
                print(f"  {key:26s} PPL {ppl:9.4f} ({ppl-base_ppl:+8.4f})  KL {kl:.5f}  "
                      f"rmse {rmse:.4f}  top1 {top1:.4f}  rel2 {stream_rel2:.5f}  [{time.time()-t0:.0f}s]",
                      flush=True)
                del lg, act, p, q
                gc.collect(); torch.cuda.empty_cache()
                json.dump(res, open(os.path.join(RES, "phase2_ablation.json"), "w"), indent=1)

    # wanda (weight pruning of the same projections; sparsity 50% == tau 0.50)
    S.CFG.on = False
    S.ST.clear()
    cdiag = {id(m.w2 if nm.endswith("ffn") else m.out_proj):
             cal[nm]["E_native"].mean(0).repeat_interleave(S.DB) for nm, m in allmods}
    for mode in ("ffn", "conv", "both"):
        act_mods = {"ffn": ffn, "conv": conv, "both": allmods}[mode]
        S.ST.clear()
        saved = wanda_prune(act_mods, cdiag)
        nll, n, _ = stream_nll(model, chunks)
        ppl = float(torch.exp(torch.tensor(nll / n)))
        lg, _ = probe_forward(model, probe, allmods, want=False)
        p, q = F.log_softmax(lg0, -1), F.log_softmax(lg, -1)
        key = f"{mode}|wanda|0.50"
        res["runs"][key] = {"ppl": ppl, "dppl": ppl - base_ppl,
                            "kl": float(F.kl_div(p, q, log_target=True, reduction="batchmean").item()),
                            "rmse": float((lg - lg0).pow(2).mean().sqrt()),
                            "top1": float((lg.argmax(-1) == lg0.argmax(-1)).float().mean()),
                            "stream_rel2": None}
        print(f"  {key:26s} PPL {ppl:9.4f} ({ppl-base_ppl:+8.4f})  [{time.time()-t0:.0f}s]", flush=True)
        restore(saved)
        del saved
        gc.collect(); torch.cuda.empty_cache()
        json.dump(res, open(os.path.join(RES, "phase2_ablation.json"), "w"), indent=1)

    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
