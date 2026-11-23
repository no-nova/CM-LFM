#!/usr/bin/env python3
"""Phase 2, revision R3.

Carried in from the review of R2:
 1. per-token RMS block energy (T-independent threshold),
 2. alpha = 1 (the sandwich Lfm3RMSNorm annihilates a per-token scalar),
 3. output-norm weighted energy and clustering:
    E[t,b] = sqrt(mean_{i in b} (a_i * ||W_out[:,i]||_2)^2), with the same static
    weights applied to the covariance before co-activation clustering.

Memory discipline: covariance is d^2 fp32, so modules are processed in batches of
<=3 (FFN) / <=4 (conv); sampled activation rows are capped at 768 per module in
fp16 and dropped as soon as the thresholds are built.

Outputs: results/phase2_calibration_r3.pt, results/phase2_r3_energy.json,
         results/phase2_r3.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, "/home/user/Downloads/CM-LFM3/src")
import lfm3_sparse as S                                     # noqa: E402

ROOT = "/home/user/Downloads/CM-LFM3"
RES = f"{ROOT}/results"
M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
DB = S.DB
TAUS = [0.10, 0.20, 0.30, 0.40, 0.50]
EVAL_START = 32768
N_CHUNK, CHUNK = 8, 1024
PROBE = 128
D = {"ffn": 7680, "conv": 2048}
GROUPS = {"rms_native": ("ident", False), "wn_native": ("ident", True),
          "rms_coact": ("pi_coact", False), "wn_coact": ("pi_wn", True)}


def registry(model):
    lm = model.model.language_model
    ffn = [(f"L{i:02d}.ffn", l.feed_forward) for i, l in enumerate(lm.layers)]
    conv = [(f"L{i:02d}.conv", l.conv) for i, l in enumerate(lm.layers) if hasattr(l, "conv")]
    return ffn, conv


def out_weights(path):
    pref = "model.language_model.layers."
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    names = {}
    for k in keys:
        if not k.startswith(pref):
            continue
        i = k[len(pref):].split(".")[0]
        if k.endswith("feed_forward.w2.weight"):
            names[f"L{int(i):02d}.ffn"] = k
        elif k.endswith("conv.out_proj.weight"):
            names[f"L{int(i):02d}.conv"] = k
    w = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for nm, k in names.items():
            W = f.get_tensor(k).float()
            w[nm] = W.pow(2).sum(0).sqrt()
            del W
    return w


def energies(rows, pi, w=None):
    ids = pi.cpu()
    a = rows.float().index_select(-1, ids)
    if w is not None:
        a = a * w.cpu().index_select(0, ids)
    nb = a.shape[-1] // DB
    return a.reshape(-1, nb, DB).pow(2).mean(-1).sqrt()


def theta_of(e, tau):
    """Threshold at the tau-quantile: drop the coldest tau fraction of blocks,
    keep (1-tau), so the retained energy is >= 1-tau for non-negative energies.
    (Using 1-tau here keeps only the *hottest* tau fraction -- the inversion the
    C++ kernel hit earlier.)"""
    return float(torch.quantile(e.reshape(-1), tau, interpolation="linear"))


def retained(e, theta):
    keep = e > theta
    return float((e * keep).sum() / e.sum()), float(keep.float().mean())


@torch.inference_mode()
def stream_nll(model, chunks):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    tot, n, prev = 0.0, 0, None
    for ch in chunks:
        out = model(input_ids=ch, past_key_values=cache, use_cache=True)
        lg = out.logits.float()[0]
        if prev is None:
            rows, tgt = lg[:-1], ch[0, 1:]
        else:
            rows = torch.cat([prev.unsqueeze(0), lg[:-1]], 0)
            tgt = ch[0]
        tot += F.cross_entropy(rows, tgt, reduction="sum").item()
        n += tgt.numel()
        prev = lg[-1].clone()
        del out, lg, rows
        gc.collect(); torch.cuda.empty_cache()
    return tot, n


def build_p3(tok, ids):
    """Chat-conditioned windows: the assistant must reproduce the text, so the
    scored span is in-distribution for this instruct-tuned backbone."""
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
        torch.cuda.empty_cache()
    return float(torch.exp(torch.tensor(tot / n))), n


@torch.inference_mode()
def probe_logits(model, ids):
    return model(input_ids=ids).logits.float()[0].cpu()


def main() -> int:
    t0 = time.time()
    S.CFG.on = False
    S.CFG.alpha = False
    S.CFG.rms = True
    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    ids = tok(open(f"{RES}/wikitext2_test.txt").read()[:400000],
              return_tensors="pt", add_special_tokens=False).input_ids[0]
    cal_chunks = [ids[i:i + 128].unsqueeze(0).cuda() for i in range(0, EVAL_START, 128)]
    seq = ids[EVAL_START:EVAL_START + N_CHUNK * CHUNK]
    chunks = [seq[i * CHUNK:(i + 1) * CHUNK].unsqueeze(0).cuda() for i in range(N_CHUNK)]
    probe = ids[EVAL_START - PROBE:EVAL_START].unsqueeze(0).cuda()
    wins = build_p3(tok, ids)

    model = AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    ffn, conv = registry(model)
    S.patch(type(ffn[0][1]), type(conv[0][1]))
    allmods = ffn + conv
    W = out_weights(f"{M}/model.safetensors")
    print(f"[{time.time()-t0:5.0f}s] model + column norms", flush=True)

    cal, audit = {}, {}
    for kind, mods, batch in (("ffn", ffn, 2), ("conv", conv, 3)):
        d = D[kind]
        for j in range(0, len(mods), batch):
            grp = mods[j:j + batch]
            S.reset_rec([m for _, m in grp], cov=True, rows=True, cap=768, dim=d)
            torch.cuda.empty_cache()
            for ch in cal_chunks:
                # logits are irrelevant here; keeping only one column of them
                # removes ~500 MB of (T, 128000) activations per chunk.
                # inference_mode is mandatory: without it every chunk's graph is
                # retained and the calibrator OOMs on the second forward.
                with torch.inference_mode():
                    model(input_ids=ch, logits_to_keep=1)
            for nm, m in grp:
                rec = S.REC[id(m)]
                c = rec["cov"]
                rows = torch.cat(rec["rows"], 0)
                w = W[nm].cuda()
                cw = c * w.unsqueeze(0) * w.unsqueeze(1)
                pi_coact, _ = S.build_perm_coact(c, d // DB)
                pi_wn, _ = S.build_perm_coact(cw, d // DB)
                del cw, c
                cal[nm] = {"d": d, "nb": d // DB, "w": W[nm], "pi_coact": pi_coact.cpu(),
                           "pi_wn": pi_wn.cpu()}
                if not torch.isfinite(rows).all():
                    print(f"  !! {nm}: {int((~torch.isfinite(rows)).sum())} non-finite lanes",
                          flush=True)
                for key, (pi_key, use_w) in GROUPS.items():
                    pi = torch.arange(d) if pi_key == "ident" else cal[nm][pi_key].long()
                    cal[nm][f"E_{key}"] = energies(rows, pi, w if use_w else None)
                del rows, w, rec
                S.REC[id(m)] = {}
            gc.collect(); torch.cuda.empty_cache()
            print(f"[{time.time()-t0:5.0f}s] {kind} batch {j//batch + 1}", flush=True)

    # thresholds + invariants
    for nm in cal:
        c = cal[nm]
        rec = {}
        for key in GROUPS:
            rec[key] = {}
            for tau in TAUS:
                th = theta_of(c[f"E_{key}"], tau)
                rho, act = retained(c[f"E_{key}"], th)
                assert rho > 1.0 - tau, f"{nm} {key} tau={tau}: retained {rho} <= {1-tau}"
                rec[key][f"{tau:.2f}"] = {"theta": th, "retained": rho, "active": act}
                c[f"theta_{key}_{tau:.2f}"] = th
        for tau in TAUS:
            t = f"{tau:.2f}"
            assert rec["wn_coact"][t]["retained"] >= rec["rms_native"][t]["retained"] - 1e-6, \
                f"{nm}: weighted/coact retains less than plain native"
        audit[nm] = rec
    json.dump(audit, open(f"{RES}/phase2_r3_energy.json", "w"), indent=1)

    nll, n = stream_nll(model, chunks)
    base = float(torch.exp(torch.tensor(nll / n)))
    lg0 = probe_logits(model, probe)
    p3b, n3 = p3_ppl(model, wins)
    print(f"patched dense: P2 PPL {base:.4f} ({n} tok)  P3 PPL {p3b:.4f} ({n3} tok)", flush=True)

    pi_of = {"rms_coact": "pi_coact", "wn_coact": "pi_wn"}
    methods = [("native_topk", "rms_native", "topk"), ("cats", "rms_coact", "topk"),
               ("ours_rms", "rms_coact", "theta"), ("ours_wn", "wn_coact", "theta"),
               ("native_wn", "wn_native", "theta")]
    out = {"dense": {"ppl": base, "n": n, "p3": p3b, "n3": n3}, "runs": {}}
    for mode in ("ffn", "conv", "both"):
        mods = {"ffn": ffn, "conv": conv, "both": allmods}[mode]
        S.CFG.mode = mode
        S.CFG.on = True
        S.CFG.pt = False
        for meth, key, rule in methods[:2] if os.environ.get("SMOKE") else methods:
            for tau in TAUS[-1:] if os.environ.get("SMOKE") else TAUS:
                S.ST.clear()          # never leave a stale gate on an inactive mode
                for nm, m in mods:
                    c = cal[nm]
                    pi = (torch.arange(c["d"], device="cuda") if key in ("rms_native", "wn_native")
                          else c[pi_of[key]].long().cuda())
                    S.set_perm(m, pi)
                    S.ST[id(m)]["theta"] = {tau: c[f"theta_{key}_{tau:.2f}"]}
                    S.ST[id(m)]["wperm"] = c["w"].cuda() if key.startswith("wn") else None
                S.CFG.method = "ours" if rule == "theta" else "topk"
                S.CFG.wnorm = key.startswith("wn")
                S.CFG.tau = tau
                nll, n = stream_nll(model, chunks)
                ppl = float(torch.exp(torch.tensor(nll / n)))
                p3, _ = p3_ppl(model, wins)
                lg = probe_logits(model, probe)
                p, q = F.log_softmax(lg0, -1), F.log_softmax(lg, -1)
                kl = float(F.kl_div(p, q, log_target=True, reduction="batchmean").item())
                top1 = float((lg.argmax(-1) == lg0.argmax(-1)).float().mean())
                out["runs"][f"{mode}|{meth}|{tau:.2f}"] = {
                    "ppl": ppl, "dppl": ppl - base, "p3": p3, "dp3": p3 - p3b,
                    "kl": kl, "top1": top1}
                print(f"  {mode:4s} {meth:12s} tau={tau:.2f} P2 {ppl:8.3f} ({ppl-base:+7.3f})"
                      f"  P3 {p3:8.3f} ({p3-p3b:+7.3f})  KL {kl:8.5f} top1 {top1:.4f}"
                      f" [{time.time()-t0:.0f}s]", flush=True)
                del lg, p, q
                gc.collect(); torch.cuda.empty_cache()
        json.dump(out, open(f"{RES}/phase2_r3.json", "w"), indent=1)

    # wanda reference on the same projections (2:4 in the gated input dim)
    S.CFG.on = False
    S.ST.clear()
    saved = []
    for nm, m in allmods:
        lin = m.w2 if nm.endswith("ffn") else m.out_proj
        wv = lin.weight.data
        imp = wv.float().abs() * cal[nm]["E_rms_native"].pow(2).mean(0).repeat_interleave(DB).sqrt().cuda()
        g = imp.reshape(wv.shape[0], wv.shape[1] // 4, 4)
        idx = g.topk(2, dim=-1).indices
        mask = torch.zeros_like(g).scatter_(-1, idx, 1.0).reshape(wv.shape)
        saved.append((lin, wv))
        lin.weight.data = (wv.float() * mask).to(wv.dtype)
    nll, n = stream_nll(model, chunks)
    ppl = float(torch.exp(torch.tensor(nll / n)))
    p3w, _ = p3_ppl(model, wins)
    out["runs"]["both|wanda|0.50"] = {"ppl": ppl, "dppl": ppl - base,
                                      "p3": p3w, "dp3": p3w - p3b}
    for lin, wv in saved:
        lin.weight.data = wv
    json.dump(out, open(f"{RES}/phase2_r3.json", "w"), indent=1)
    torch.save({nm: {k: v for k, v in cal[nm].items() if not k.startswith("E_")} for nm in cal},
               f"{RES}/phase2_calibration_r3.pt")
    print(f"wanda PPL {ppl:.3f} | done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
