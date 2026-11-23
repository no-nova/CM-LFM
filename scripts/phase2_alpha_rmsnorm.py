"""Bug 1b: does a scalar alpha survive the sandwich RMSNorm?

The Phase-2 alpha rescaling multiplies the gated activation inside the projection.
`modeling_lfm3.py` then applies `post_ffn_norm = Lfm3RMSNorm(d)` to the projection
output, and RMSNorm divides by the per-token RMS across all d channels -- so a
per-token scalar must cancel up to the eps term.

This script measures that cancellation directly on real weights and real activations,
for the FFN and for LFM3Conv (post_operator_norm), at tau=0.50:
  * |y_masked(alpha=1.5) - y_masked(alpha=1)| before the norm  (the intended effect)
  * the same after the norm                                   (what survives)
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODEL = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
DB = 32


def rms_norm(x, w, eps=1e-5):
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main() -> int:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from transformers import AutoModelForImageTextToText

    cal = torch.load(os.path.join(RES, "phase2_calibration.pt"), map_location="cpu", weights_only=False)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(MODEL)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        attn_implementation="sdpa", trust_remote_code=True).eval()
    lm = model.model.language_model

    captured = {}

    def mk_hook(name):
        def hook(_m, args, kwargs):
            x = kwargs.get("hidden_states", args[0] if args else None)
            if x is not None and name not in captured:
                captured[name] = x.detach().float().cpu()
        return hook

    hs = lm.layers[0]
    hs.feed_forward.register_forward_pre_hook(mk_hook("ffn"), with_kwargs=True)
    hs.conv.register_forward_pre_hook(mk_hook("conv"), with_kwargs=True)
    enc = tok("The history of the Roman Republic is", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.model(**enc)
    del model
    torch.cuda.empty_cache()

    out = {}
    with safe_open(os.path.join(MODEL, "model.safetensors"), framework="pt", device="cpu") as f:
        for fam, prefix, cap in (
            ("ffn", "model.language_model.layers.0.feed_forward", "ffn"),
            ("conv", "model.language_model.layers.0.conv", "conv"),
        ):
            w = {k: f.get_tensor(f"{prefix}.{k}.weight").float() for k in
                 ("w1", "w3", "w2") if f"{prefix}.{k}.weight" in f.keys()}
            if fam == "ffn":
                pn = f.get_tensor(f"model.language_model.layers.0.post_ffn_norm.weight").float()
            else:
                w = {k.split(".")[0]: f.get_tensor(f"{prefix}.{k}").float()
                     for k in ("in_proj.weight", "out_proj.weight", "conv.weight")}
                pn = f.get_tensor("model.language_model.layers.0.post_operator_norm.weight").float()

            x = captured[cap].reshape(-1, 2048)
            key = "L00.ffn" if fam == "ffn" else "L00.conv"
            pi = cal[key]["pi_coact"]
            th = cal[key]["theta_coact"][0.50]
            nb = (7680 if fam == "ffn" else 2048) // DB

            if fam == "ffn":
                h = F.silu(x @ w["w1"].T) * (x @ w["w3"].T)          # (n, 7680)
                hp = h.index_select(-1, pi)
                e = hp.reshape(-1, nb, DB).pow(2).sum(-1)
                keep = (e > th)
                m = (keep.repeat_interleave(DB, -1).float())
                dense = h @ w["w2"].T
                r = {}
                for a in (1.0, 1.25, 1.5):
                    a_tok = (e.sum(-1).sqrt() / ((e * keep).sum(-1)).sqrt().clamp_min(1e-12)).clamp_max(1.5) if a > 1 else torch.ones(x.shape[0])
                    scale = a_tok * a
                    y = (h * (m * scale[:, None])) @ w["w2"].T
                    r[f"alpha_{a}"] = y
            else:
                xc = x.T.unsqueeze(0)                                 # (1, d, T)
                BCX = (w["in_proj"] @ xc[0]).chunk(3, 0)
                B, C, X = BCX
                hmix = B * X
                kk = w["conv"].shape[-1]
                yc = F.conv1d(F.pad(hmix.unsqueeze(0), (kk - 1, 0)), w["conv"], None, groups=2048)[..., :hmix.shape[-1]]
                y = (C * yc[0])                                       # (d, T)
                yp = y.t()
                e = yp.index_select(-1, pi).reshape(-1, nb, DB).pow(2).sum(-1)
                keep = (e > th)
                m = keep.repeat_interleave(DB, -1).float()
                dense = (w["out_proj"] @ y).t()
                r = {}
                for a in (1.0, 1.25, 1.5):
                    a_tok = (e.sum(-1).sqrt() / ((e * keep).sum(-1)).sqrt().clamp_min(1e-12)).clamp_max(1.5) if a > 1 else torch.ones(x.shape[0])
                    scale = a_tok * a
                    ym = (w["out_proj"] @ (y * (m.T * scale[None, :])))
                    r[f"alpha_{a}"] = ym.t()

            rec = {"n_tokens": int(x.shape[0]), "retained_energy": float((e * keep).sum() / e.sum()),
                   "post_norm": "post_ffn_norm" if fam == "ffn" else "post_operator_norm"}
            y0 = r["alpha_1.0"]
            y0n = rms_norm(y0, pn)
            for a in (1.25, 1.5):
                ya, yan = r[f"alpha_{a}"], rms_norm(r[f"alpha_{a}"], pn)
                rec[f"rel_alpha{a}_pre_norm"] = rel(ya, y0)
                rec[f"rel_alpha{a}_post_norm"] = rel(yan, y0n)
                rec[f"rel_alpha{a}_vs_dense_post_norm"] = rel(yan, rms_norm(dense, pn))
            rec["rel_masked_vs_dense_pre_norm"] = rel(y0, dense)
            rec["rel_masked_vs_dense_post_norm"] = rel(y0n, rms_norm(dense, pn))
            # why the cancellation is not complete: RMSNorm's eps sets a floor on the
            # token variance, and below that floor the norm stops being scale-invariant.
            var = y0.pow(2).mean(-1)
            eps = 1e-5
            s = 1.5
            surv = s * (var + eps).sqrt() / (s * s * var + eps).sqrt()
            rec["token_var_median"] = float(var.median())
            rec["eps_over_var_median"] = float(eps / var.median())
            rec["analytic_scale_survival_s1.5"] = float(surv.median())
            out[fam] = rec
            print(fam, json.dumps(rec, indent=1))

    with open(os.path.join(RES, "phase2_alpha_rmsnorm.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
