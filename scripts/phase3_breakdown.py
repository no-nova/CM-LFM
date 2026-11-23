#!/usr/bin/env python3
"""Phase 3 (R3): where CPU decode time actually goes, and what quantisation buys.

Runs the real LFM3-VL-2B language backbone on the CPU with the zero-copy mmap
loader (anonymous RSS stays ~0.5 GB) and derives the exclusive cost of each
operator class by ablation: measure decode, replace every FFN (resp. conv,
attention) submodule with Identity, measure again, and take the difference.
Per-module hooks were tried first and returned 4 us per projection (the deltas
were lost across nested calls), so the ablation method is used instead.  Then
the *measured* IQ4_NL block-major kernel times are substituted for each
projection and the composed end-to-end decode latency is reported.

Two estimates are produced:
  conservative  t_dense_component * (kernel_iq4 / kernel_bf16)   keeps the
                in-model inefficiency of the dense path;
  composed      sum(measured kernel_iq4) + measured non-GEMM time (norms, rope,
                softmax, depthwise conv, embedding, python).
The second is the honest projection of what the fused CPU path would cost, the
first is the pessimistic bound.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/user/Downloads/CM-LFM3/src")
ROOT = "/home/user/Downloads/CM-LFM3"
RES = f"{ROOT}/results"
MODEL = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"

# measured kernel times at T=1, 16 threads, DRAM-cold (artifacts/p3k, artifacts/p3gqa)
K = {
    "ffn":   {"bf16": 2.0309, "iq4": 0.7861},   # w1/w3/w2 with tau=0.5 block skip
    "conv":  {"bf16": 0.6734, "iq4": 0.1655},   # in_proj + dwconv + out_proj, tau=0.5
    "q":     {"bf16": 0.2313, "iq4": 0.1129},
    "k":     {"bf16": 0.0633, "iq4": 0.0272},
    "v":     {"bf16": 0.0566, "iq4": 0.0282},
    "o":     {"bf16": 0.2315, "iq4": 0.1027},
    "lm":    {"bf16": 14.5098, "iq4": 6.5765},
}


def rss():
    d = {}
    for ln in open("/proc/self/status"):
        if ln.startswith(("RssAnon", "RssFile", "VmRSS")):
            k, v = ln.split(":")
            d[k] = int(v.split()[0]) / 1024.0
    return {k: round(v, 1) for k, v in d.items()}


def mmap_model():
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
    st = {}
    with safe_open(os.path.join(MODEL, "model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            st[k] = f.get_tensor(k)
    model.load_state_dict(st, strict=False, assign=True)
    del st
    gc.collect()
    model.tie_weights()
    for name, m in model.named_modules():
        if type(m).__name__.endswith("RotaryEmbedding") and any(b.is_meta for b in m.buffers()):
            cfgm = cfg.vision_config if ".vision" in name else cfg.text_config
            type(m).__init__(m, cfgm)
    return model.eval()


class Skip(torch.nn.Module):
    """Identity that tolerates the keyword-only calling convention of LFM3 blocks."""

    def forward(self, hidden_states=None, *a, **k):
        return hidden_states


class SkipAttn(Skip):
    """The attention block returns (output, weights), so the stub must match."""

    def forward(self, hidden_states=None, *a, **k):
        return hidden_states, None


def decode_ms(lm, model, ids, n=6):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        lm(input_ids=ids, past_key_values=cache, use_cache=True)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            lm(input_ids=torch.randint(0, 1000, (1, 1)), past_key_values=cache, use_cache=True)
            ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2]


def head_ms(lm_head, n=5):
    h = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    with torch.inference_mode():
        for _ in range(2):
            lm_head(h)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            lm_head(h)
            ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main() -> int:
    torch.set_num_threads(16)
    model = mmap_model()
    lm = model.model.language_model
    ids = torch.randint(0, 1000, (1, 8))
    total = decode_ms(lm, model, ids)
    t_lm = head_ms(model.get_output_embeddings())

    global SKIP
    SKIP = Skip()
    SKIP_ATTN = SkipAttn()
    cats = {"ffn": ("feed_forward",), "conv": ("conv",), "attn": ("self_attn",)}
    saved, delta = {}, {}
    for cat, leaf in cats.items():
        hit = [l for l in lm.layers if any(hasattr(l, a) and getattr(l, a) is not None for a in leaf)]
        held = []
        for l in hit:
            for a in leaf:
                if getattr(l, a, None) is not None:
                    held.append((l, a, getattr(l, a)))
                    setattr(l, a, SKIP_ATTN if a == "self_attn" else SKIP)
        delta[cat] = total - decode_ms(lm, model, ids)
        for l, a, m in held:
            setattr(l, a, m)
    # direct measurement of everything that is not a projection: stub all three
    held = []
    for l in lm.layers:
        for a in ("feed_forward", "conv", "self_attn"):
            if getattr(l, a, None) is not None:
                held.append((l, a, getattr(l, a)))
                setattr(l, a, SKIP_ATTN if a == "self_attn" else SKIP)
    t_other_all = decode_ms(lm, model, ids)
    for l, a, m in held:
        setattr(l, a, m)
    n_layers, n_conv = len(lm.layers), sum(1 for l in lm.layers if hasattr(l, "conv"))
    n_attn = n_layers - n_conv
    gemm_dense = sum(delta.values()) + t_lm
    nongemm = t_other_all          # measured directly, no negative subtraction
    ratio = {"ffn": K["ffn"]["iq4"] / K["ffn"]["bf16"],
             "conv": K["conv"]["iq4"] / K["conv"]["bf16"],
             "gqa": (K["q"]["iq4"] + K["k"]["iq4"] + K["v"]["iq4"] + K["o"]["iq4"]) /
                    (K["q"]["bf16"] + K["k"]["bf16"] + K["v"]["bf16"] + K["o"]["bf16"]),
             "lm": K["lm"]["iq4"] / K["lm"]["bf16"]}
    t_cons = (delta["ffn"] * ratio["ffn"] + delta["conv"] * ratio["conv"] +
              delta["attn"] * ratio["gqa"] + t_lm * ratio["lm"] + nongemm)
    t_gemm_new = (K["ffn"]["iq4"] * n_layers + K["conv"]["iq4"] * n_conv +
                  (K["q"]["iq4"] + K["k"]["iq4"] + K["v"]["iq4"] + K["o"]["iq4"]) * n_attn +
                  K["lm"]["iq4"])
    t_comp = t_gemm_new + nongemm
    out = {
        "threads": 16, "layers": n_layers, "conv_layers": n_conv, "attn_layers": n_attn,
        "method": "exclusive operator cost by module ablation (Identity swap)",
        "decode_ms_per_token_dense": round(total, 2),
        "components_ms": {"ffn_all": round(delta["ffn"], 2), "conv_all": round(delta["conv"], 2),
                          "attn_all": round(delta["attn"], 2), "lm_head": round(t_lm, 2)},
        "components_pct": {k: round(100 * v / total, 1) for k, v in
                           [("ffn", delta["ffn"]), ("conv", delta["conv"]),
                            ("attn", delta["attn"]), ("lm_head", t_lm),
                            ("other", nongemm)]},
        "gemm_dense_ms": round(gemm_dense, 2), "other_ms": round(nongemm, 2),
        "ablation_overlap_ms": round(gemm_dense - (total - t_other_all), 2),
        "other_pct": round(100 * t_other_all / total, 1),
        "kernel_times_ms": K, "kernel_ratio": {k: round(v, 3) for k, v in ratio.items()},
        "composed_iq4_gemm_ms": round(t_gemm_new, 2),
        "e2e_conservative_ms": round(t_cons, 2),
        "e2e_conservative_speedup": round(total / t_cons, 2),
        "e2e_composed_ms": round(t_comp, 2),
        "e2e_composed_speedup": round(total / t_comp, 2),
        "weight_bytes_dense_MB": 3658.6,
        "weight_bytes_iq4_skip_MB": round(2160.0 * 18 / 32 * 0.5 + 416.4 * 18 / 32 * 0.5 +
                                          220.7 * 18 / 32 + 500.0 * 18 / 32, 1),
        "rss": rss(),
        "other_note": "norms, rope, softmax, depthwise conv1d, cache updates, embedding, dispatch",
    }
    json.dump(out, open(f"{RES}/phase3_e2e_r3.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:32s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
