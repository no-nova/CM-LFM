#!/usr/bin/env python3
"""Diagnose why the bf16 whole-model pi_ffn test moved 0.098 in RMSE.

Controls:
  c0 determinism        : same weights, two forwards
  c1 identity permutation: build new (copied) weight tensors with pi = arange
  c2 reduced-precision bf16 GEMM reduction ON (torch default) vs OFF
  c3 single-layer permutation (layer 0 only) vs all layers
"""
import json, os
import torch
from transformers import AutoModelForImageTextToText
M="/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"; RES="/home/user/Downloads/CM-LFM3/results"

def load():
    return AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()

@torch.inference_mode()
def run(model, ids, perm_fn=None, layers=None, pi=None, flag=None):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    if perm_fn: perm_fn(model, layers, pi)
    return model(input_ids=ids).logits.float()

@torch.inference_mode()
def main():
    model = load()
    lm = model.model.language_model
    ids = torch.randint(0, 120000, (1, 128), device="cuda")
    res = {}
    for flag in (True, False):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        base = model(input_ids=ids).logits.float()
        again = model(input_ids=ids).logits.float()
        res[f"determinism_flag{flag}"] = float((base - again).abs().max())
        ffn = [getattr(l, "mlp", None) or getattr(l, "feed_forward", None) for l in lm.layers]
        ffn = [f for f in ffn if f is not None]
        # identity permutation (new tensors, same values)
        def idperm(m, layers, pi):
            for l in ffn:
                f = l
                q = torch.arange(f.w1.out_features, device="cuda")
                f.w1.weight = torch.nn.Parameter(f.w1.weight[q])
                f.w3.weight = torch.nn.Parameter(f.w3.weight[q])
                f.w2.weight = torch.nn.Parameter(f.w2.weight[:, q])
        res[f"identity_perm_flag{flag}"] = float((run(model, ids, idperm, flag=flag) - base).abs().max())
        # real permutation, all layers
        def realperm(m, layers, pi):
            for l in ffn:
                f = l
                f.w1.weight = torch.nn.Parameter(f.w1.weight[pi])
                f.w3.weight = torch.nn.Parameter(f.w3.weight[pi])
                f.w2.weight = torch.nn.Parameter(f.w2.weight[:, pi])
        g = torch.Generator(device="cuda").manual_seed(3)
        pi = torch.randperm(7680, generator=g, device="cuda")
        res[f"pi_ffn_all_flag{flag}"] = float((run(model, ids, realperm, pi=pi, flag=flag) - base).abs().max())
        # single layer only
        def oneperm(m, layers, pi):
            f = ffn[0]
            f.w1.weight = torch.nn.Parameter(f.w1.weight[pi])
            f.w3.weight = torch.nn.Parameter(f.w3.weight[pi])
            f.w2.weight = torch.nn.Parameter(f.w2.weight[:, pi])
        res[f"pi_ffn_layer0_flag{flag}"] = float((run(model, ids, oneperm, pi=pi, flag=flag) - base).abs().max())
        # restore (undo all) by reloading is cheapest
        model = load()
        lm = model.model.language_model
        ffn = [getattr(l, "mlp", None) or getattr(l, "feed_forward", None) for l in lm.layers]
        ffn = [f for f in ffn if f is not None]
        clear = float((model(input_ids=ids).logits.float() - base).abs().max())
        res[f"reload_bitwise_identical_flag{flag}"] = clear
    json.dump(res, open(os.path.join(RES, "phase2_iso_diag.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

main()
