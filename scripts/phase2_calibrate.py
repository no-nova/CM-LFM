#!/usr/bin/env python3
"""Phase 2 calibration: activation covariance -> pi (Gram-greedy / co-activation)
and per-layer block-energy thresholds theta_l(tau).

Calibration split: 128 samples x 256 tokens = 32,768 WikiText-2 tokens taken
from the FIRST 32k tokens; perplexity evaluation starts after them, so
calibration and evaluation text do not overlap.
Output: results/phase2_calibration.pt, results/phase2_cov.pt,
        results/phase2_energy_diag.json
"""
import gc, json, os, sys, time
import torch

sys.path.insert(0, "/home/user/Downloads/CM-LFM3/src")
import lfm3_sparse as S                                    # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
RES = "/home/user/Downloads/CM-LFM3/results"
TAUS = [0.10, 0.20, 0.30, 0.40, 0.50]
N_SAMP, SAMP_TOK = 128, 256
D_FFN, D_CONV = 7680, 2048


def registry(model):
    lm = model.model.language_model
    ffn = [(f"L{i:02d}.ffn", l.feed_forward) for i, l in enumerate(lm.layers)]
    conv = [(f"L{i:02d}.conv", l.conv) for i, l in enumerate(lm.layers) if hasattr(l, "conv")]
    return ffn, conv


@torch.inference_mode()
def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    text = open(os.path.join(RES, "wikitext2_test.txt")).read()[:200000]
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    batches = [ids[i * SAMP_TOK:(i + 1) * SAMP_TOK].unsqueeze(0).cuda() for i in range(N_SAMP)]
    model = AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    ffn, conv = registry(model)
    S.patch(type(ffn[0][1]), type(conv[0][1]))
    print(f"registry {len(ffn)} ffn (d={D_FFN}) + {len(conv)} conv (d={D_CONV}); "
          f"calib tokens {sum(b.shape[1] for b in batches)}; t={time.time()-t0:.0f}s", flush=True)

    allmods = [(n, m, D_FFN) for n, m in ffn] + [(n, m, D_CONV) for n, m in conv]
    # identity permutations for the covariance/energy passes
    for _, m, d in allmods:
        S.set_perm(m, S.identity(d))
    name_of = {id(m): n for n, m, _ in allmods}

    # ---------------- covariance: FFN 3 layers at a time, all 13 conv at once
    cov = {}
    groups = [ffn[i:i + 3] for i in range(0, len(ffn), 3)] + [conv]
    for gi, grp in enumerate(groups):
        dim = D_FFN if gi < len(groups) - 1 else D_CONV
        S.reset_rec([m for _, m in grp], cov=True, dim=dim)
        for b in batches:
            model(input_ids=b)
        for n, m in grp:
            cov[n] = S.REC[id(m)]["cov"].cpu()
            del S.REC[id(m)]
        gc.collect(); torch.cuda.empty_cache()
        print(f"  C[{grp[0][0]}..{grp[-1][0]}] {time.time()-t0:.0f}s", flush=True)
    torch.save(cov, os.path.join(RES, "phase2_cov.pt"))
    print(f"  covariance saved ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- permutations
    out = {}
    for n, m, d in allmods:
        C = cov[n].cuda()
        W = (m.w2.weight if d == D_FFN else m.out_proj.weight).detach().float()
        nb = d // S.DB
        out[n] = {}
        for kind, fn in (("gram", S.build_perm_gram), ("coact", S.build_perm_coact)):
            pi, inv = fn(C, W, nb) if kind == "gram" else fn(C, nb)
            out[n][f"pi_{kind}"] = pi.cpu().to(torch.int32)
            out[n][f"inv_{kind}"] = inv.cpu().to(torch.int32)
        del C
        gc.collect(); torch.cuda.empty_cache()
    del cov
    gc.collect(); torch.cuda.empty_cache()
    print(f"  permutations done {time.time()-t0:.0f}s", flush=True)

    # ---------------- block-energy distributions under each order
    for kind in ("native", "gram", "coact"):
        for n, m, d in allmods:
            pi = S.identity(d) if kind == "native" else out[n][f"pi_{kind}"].long().cuda()
            S.set_perm(m, pi)
        S.reset_rec([m for _, m, _ in allmods], energy=True, cap=4096)
        for b in batches:
            model(input_ids=b)
        for n, m, d in allmods:
            e = S.rec_rows(S.REC[id(m)])
            out[n][f"E_{kind}"] = e
            out[n][f"theta_{kind}"] = {t: S.theta_for(t, [e]) for t in TAUS}
            del S.REC[id(m)]
        print(f"  energies[{kind}] done {time.time()-t0:.0f}s", flush=True)

    torch.save(out, os.path.join(RES, "phase2_calibration.pt"))

    # ---------------- mechanism diagnostics: dropped energy fraction per order
    diag = {}
    for n, _, _ in allmods:
        row = {}
        for kind in ("native", "gram", "coact"):
            e2 = out[n][f"E_{kind}"]          # squared block energy (already squared)
            tot = e2.sum(-1).clamp_min(1e-12)
            row[kind] = {}
            for t in TAUS:
                k = int(round((1 - t) * e2.shape[-1]))
                row[kind][t] = float((1 - e2.topk(k, dim=-1).values.sum(-1) / tot).mean())
        diag[n] = row
    json.dump(diag, open(os.path.join(RES, "phase2_energy_diag.json"), "w"), indent=1)
    print("\nmean dropped-energy fraction across text FFN layers (native/gram/coact):")
    for t in TAUS:
        v = [[diag[n][k][t] for k in ("native", "gram", "coact")] for n, _, d in allmods if d == D_FFN]
        m3 = [sum(x[i] for x in v) / len(v) for i in range(3)]
        print(f"  tau={t:.2f}: {m3[0]:.4f} / {m3[1]:.4f} / {m3[2]:.4f}")
    print("\nmean dropped-energy fraction across LFM3Conv layers (native/gram/coact):")
    for t in TAUS:
        v = [[diag[n][k][t] for k in ("native", "gram", "coact")] for n, _, d in allmods if d == D_CONV]
        m3 = [sum(x[i] for x in v) / len(v) for i in range(3)]
        print(f"  tau={t:.2f}: {m3[0]:.4f} / {m3[1]:.4f} / {m3[2]:.4f}")
    print(f"\nDONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
