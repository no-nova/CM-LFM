#!/usr/bin/env python3
"""Dump the non-gated projections (GQA q/k/v/o, tied lm_head) for the R3 kernel.

Reads only the needed tensors straight out of the safetensors header (mmap), no
model load, so peak RSS stays in the tens of MB.  lm_head is tied to
embed_tokens, so the 500 MiB matrix is dumped once.
"""
import json
import os

import torch
from safetensors import safe_open

ROOT = "/home/user/Downloads/CM-LFM3"
M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B/model.safetensors"
OUT = f"{ROOT}/artifacts/p3"
WANT = {
    "q": "model.language_model.layers.2.self_attn.q_proj.weight",
    "k": "model.language_model.layers.2.self_attn.k_proj.weight",
    "v": "model.language_model.layers.2.self_attn.v_proj.weight",
    "o": "model.language_model.layers.2.self_attn.o_proj.weight",
    "lm": "model.language_model.embed_tokens.weight",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    with safe_open(M, framework="pt", device="cpu") as f:
        for tag, key in WANT.items():
            W = f.get_tensor(key).to(torch.bfloat16).contiguous()
            W.view(torch.uint16).numpy().tofile(f"{OUT}/p3_{tag}.bin")
            meta[tag] = {"key": key, "shape": list(W.shape)}
            print(f"{tag:3s} {tuple(W.shape)} {W.numel() * 2 / 2**20:.1f} MiB", flush=True)
            del W
    json.dump(meta, open(f"{OUT}/p3_gqa_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
