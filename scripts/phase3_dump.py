#!/usr/bin/env python3
"""Phase 3 data prep: dump one real LFM3Conv layer + real activations as raw bf16.

Everything is read with safetensors mmap (zero-copy) and only the tensors needed
for one layer are materialised, so peak anon RSS stays far below the 1.5 GB cap.
Outputs (all little-endian, row-major unless stated):
  p3_in.bin    in_proj.weight      (3d, d)   bf16
  p3_cw.bin    conv.weight         (d, 1, k) bf16  (flattened d*k)
  p3_cb.bin    conv.bias           (d)       bf16
  p3_out.bin   out_proj.weight     (d, d)    bf16
  p3_x.bin     layer input x       (T, d)    bf16
  p3_ref.bin   PyTorch expert output (T, d)  fp32
  p3_meta.json dims, theta per tau, block energies of the real activations
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import lfm3_sparse as S  # noqa: E402

M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B/model.safetensors"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "p3")
os.makedirs(OUT, exist_ok=True)
T = 64
LAYER = 0


def bf(t):
    return t.detach().to(torch.bfloat16).contiguous().cpu().view(torch.uint16).numpy().tobytes()


def main():
    # --- model weights: header only, then one layer at a time
    f = safe_open(M, framework="pt", device="cpu")
    keys = list(f.keys())
    pref = f"model.language_model.layers.{LAYER}.conv."
    w_in = f.get_tensor(pref + "in_proj.weight")
    w_cw = f.get_tensor(pref + "conv.weight").reshape(-1)
    cbk = pref + "conv.bias"
    w_cb = f.get_tensor(cbk) if cbk in keys else torch.zeros(2048, dtype=torch.bfloat16)
    w_out = f.get_tensor(pref + "out_proj.weight")
    d = w_in.shape[1]
    print(f"layer {LAYER}: in {tuple(w_in.shape)} pass conv {tuple(w_cw.shape)} out {tuple(w_out.shape)} "
          f"d={d} params={(w_in.numel()+w_cw.numel()+w_out.numel())/1e6:.2f}M")
    open(f"{OUT}/p3_in.bin", "wb").write(bf(w_in))
    open(f"{OUT}/p3_cw.bin", "wb").write(bf(w_cw))
    open(f"{OUT}/p3_cb.bin", "wb").write(bf(w_cb))
    open(f"{OUT}/p3_out.bin", "wb").write(bf(w_out))
    del w_in, w_out
    import gc
    gc.collect()

    # --- reference activations: run the real model on real text, take the input
    # of layer 0's LFM3Conv (the ffn_norm output is not what the conv sees: the
    # conv consumes operator_norm(hidden_states), so capture that tensor).
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    tok = AutoTokenizer.from_pretrained(os.path.dirname(M), trust_remote_code=True)
    ids = tok(open(f"{res}/wikitext2_test.txt").read()[:400000], return_tensors="pt",
              add_special_tokens=False).input_ids[0][32768:32768 + T].unsqueeze(0).cuda()
    model = AutoModelForImageTextToText.from_pretrained(
        os.path.dirname(M), dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    grab = {}

    def hook(mod, args, kwargs):
        grab["x"] = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]).detach()

    h = model.model.language_model.layers[LAYER].conv.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    h.remove()
    x = grab["x"]
    print("captured layer input", tuple(x.shape), "absmax", float(x.abs().max()))

    # reference forward of this one layer, copied from modeling_lfm3.py
    with torch.inference_mode():
        w = model.model.language_model.layers[LAYER].conv
        bcx = w.in_proj(x).transpose(-1, -2)
        B, C, X = bcx[:, :d], bcx[:, d:2 * d], bcx[:, 2 * d:]
        hh = B * X
        y = F.conv1d(hh, w.conv.weight, w.conv.bias, padding=w.conv.padding[0], groups=d)[..., :x.shape[1]]
        y = C * y
        ref = w.out_proj(y.transpose(-1, -2).contiguous()).float()
    open(f"{OUT}/p3_x.bin", "wb").write(bf(x))
    open(f"{OUT}/p3_ref.bin", "wb").write(ref.cpu().contiguous().numpy().astype("<f4").tobytes())

    # --- block energies of the real gated intermediate y, and calibrated theta
    yt = y.transpose(-1, -2).float()                       # (1, T, d)
    e = S.block_energy(yt, d // S.DB)[0]                   # (T, nb)
    cal = torch.load(f"{res}/phase2_calibration.pt", weights_only=False)[f"L{LAYER:02d}.conv"]
    meta = {"d": d, "T": T, "k": int(w.conv.weight.shape[-1]), "nb": d // S.DB,
            "theta": {t: float(cal["theta_native"][t]) for t in (0.20, 0.30, 0.40, 0.50)},
            "theta_gram": {t: float(cal["theta_gram"][t]) for t in (0.20, 0.30, 0.40, 0.50)},
            "block_energy_mean": e.mean(0).tolist(),
            "energy_total": float(e.sum())}
    with open(f"{OUT}/p3_meta.json", "w") as fh:
        json.dump(meta, fh)
    frac = {t: float((e < meta["theta"][t]).float().mean()) for t in meta["theta"]}
    print("real-activation block drop fractions:", {k: round(v, 4) for k, v in frac.items()})
    print("theta:", meta["theta"])

    # --- FFN layer 0 (d=2048 -> d_ff=7680) + its real input, for the Bug-3 sweep:
    # 59.04% of the checkpoint's bytes live here, so a CPU result that ignores it
    # says nothing about the model.
    pf = f"model.language_model.layers.{LAYER}.feed_forward."
    w1 = f.get_tensor(pf + "w1.weight")
    w3 = f.get_tensor(pf + "w3.weight")
    w2 = f.get_tensor(pf + "w2.weight")
    print(f"ffn  w1 {tuple(w1.shape)} w3 {tuple(w3.shape)} w2 {tuple(w2.shape)}")
    open(f"{OUT}/p3_w1.bin", "wb").write(bf(w1))
    open(f"{OUT}/p3_w3.bin", "wb").write(bf(w3))
    open(f"{OUT}/p3_w2.bin", "wb").write(bf(w2))
    del w1, w3, w2
    gc.collect()

    grab2 = {}

    def hook2(mod, args, kwargs):
        grab2["x"] = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]).detach()

    hh2 = model.model.language_model.layers[LAYER].feed_forward.register_forward_pre_hook(hook2, with_kwargs=True)
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False)
    hh2.remove()
    xf = grab2["x"]
    with torch.inference_mode():
        ff = model.model.language_model.layers[LAYER].feed_forward
        ref_f = ff(xf).float()
        hid = F.silu(ff.w1(xf)) * ff.w3(xf)
    print("captured ffn input", tuple(xf.shape), "gated", tuple(hid.shape), "absmax", float(xf.abs().max()))
    open(f"{OUT}/p3_xf.bin", "wb").write(bf(xf))
    open(f"{OUT}/p3_reff.bin", "wb").write(ref_f.cpu().contiguous().numpy().astype("<f4").tobytes())
    ef = S.block_energy(hid.float().reshape(-1, hid.shape[-1]), 240).reshape(hid.shape[0], hid.shape[1], 240)
    meta["ffn"] = {"d": 2048, "d_ff": int(hid.shape[-1]), "nb": 240, "T": T,
                   "block_energy_mean": ef.mean((0, 1)).tolist()}
    with open(f"{OUT}/p3_meta.json", "w") as fh:
        json.dump(meta, fh)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
