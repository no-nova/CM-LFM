#!/usr/bin/env python3
"""Phase 1: architecture anatomy from the safetensors *header only* (no weight data read).

Reads the 8-byte header length + JSON header of model.safetensors, so peak RSS is
O(header) ~ a few MB. Emits a per-tensor table, per-layer-type parameter/byte
breakdown, and the analytic arithmetic intensity of each operator class.
"""
import json, struct, sys, os
from collections import defaultdict, OrderedDict

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
CFG = json.load(open(os.path.join(MODEL, "config.json")))
ST = os.path.join(MODEL, "model.safetensors")

with open(ST, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

meta = hdr.pop("__metadata__", {})
DT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}

rows = []
for name, info in hdr.items():
    shape = info["shape"]
    dt = info["dtype"]
    numel = 1
    for s in shape:
        numel *= s
    nbytes = numel * DT_BYTES.get(dt, 2)
    rows.append((name, shape, dt, numel, nbytes))

total_bytes = sum(r[4] for r in rows)
total_params = sum(r[3] for r in rows)
print(f"# tensors: {len(rows)}   total elements: {total_params:,}   total bytes: {total_bytes:,} "
      f"({total_bytes/2**30:.3f} GiB)  file: {os.path.getsize(ST):,} B")
print(f"# metadata: {meta}")

# ---- per top-level module aggregation -------------------------------------
def bucket(name):
    p = name.split(".")
    if p[0] == "model" and p[1] == "language_model":
        if p[2] == "embed_tokens": return "text.embed_tokens"
        if p[2] == "norm": return "text.final_norm"
        if p[2] == "layers":
            li = int(p[3])
            lt = CFG["text_config"]["layer_types"][li]
            rest = p[4]
            if rest in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
                return f"text.layer.{lt}.norms"
            if rest == "feed_forward":
                return f"text.layer.{lt}.ffn." + p[5]
            if rest == "conv":
                return f"text.layer.conv.{p[5]}" if len(p) > 5 else f"text.layer.conv"
            if rest == "self_attn":
                leaf = p[5]
                if leaf.endswith("_norm"):
                    return f"text.layer.{lt}.attn.norms"
                return f"text.layer.{lt}.attn." + leaf
            return f"text.layer.{lt}.other"
    if p[0] == "model" and p[1] == "vision_tower":
        # vision tower
        if "layers" in p:
            li = int(p[p.index("layers") + 1])
            lt = CFG["vision_config"]["layer_types"][li]
            rest = p[p.index("layers") + 2]
            if rest == "conv":
                return f"vision.layer.conv.{p[-1]}" if p[-1] in ("in_proj.weight","out_proj.weight","conv.weight") else "vision.layer.conv"
            if rest == "self_attn":
                leaf = p[-1]
                if leaf in ("q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight"):
                    return f"vision.layer.{lt}.attn.{leaf}"
                return f"vision.layer.{lt}.attn.other"
            if rest == "mlp":
                return f"vision.layer.mlp.{p[-1]}"
            return f"vision.layer.{rest}"
        return "vision." + ".".join(p[2:])
    if p[0] == "model" and p[1] == "multi_modal_projector":
        return "mm_projector." + ".".join(p[2:])
    if p[0] == "lm_head":
        return "lm_head"
    return ".".join(p[:3])

agg = defaultdict(lambda: [0, 0, 0])
for name, shape, dt, numel, nb in rows:
    b = bucket(name)
    agg[b][0] += 1; agg[b][1] += numel; agg[b][2] += nb

print("\n## Per-component parameter / byte breakdown")
print(f"{'component':46s} {'tensors':>7s} {'params':>14s} {'MiB':>10s} {'%bytes':>7s}")
for k in sorted(agg, key=lambda k: -agg[k][2]):
    c, p, nb = agg[k]
    print(f"{k:46s} {c:7d} {p:14,d} {nb/2**20:10.2f} {100*nb/total_bytes:6.2f}%")

# ---- layer topology -------------------------------------------------------
tc = CFG["text_config"]
print("\n## Text backbone topology")
print(f"hidden d={tc['hidden_size']}  d_ff={tc['intermediate_size']}  layers={tc['num_hidden_layers']}  "
      f"heads={tc['num_attention_heads']}x{tc['head_dim']}  kv_heads={tc['num_key_value_heads']}  "
      f"vocab={tc['vocab_size']}  conv_k={tc['conv_L_cache']}  sw={tc['sliding_window']}  tie={tc['tie_word_embeddings']}")
from collections import Counter
print("text layer_types:", dict(Counter(tc["layer_types"])))
vc = CFG["vision_config"]
print(f"vision: d={vc['hidden_size']} d_ff={vc['intermediate_size']} layers={vc['num_hidden_layers']} "
      f"heads={vc['num_attention_heads']} kv={vc['num_key_value_heads']} patch={vc['patch_size']} "
      f"temporal={vc['temporal_patch_size']} merge={vc['spatial_merge_size']} conv_k={vc['conv_kernel_size']}")
print("vision layer_types:", dict(Counter(vc["layer_types"])))
print(f"projector_hidden={CFG['projector_hidden_size']}")

# ---- analytic per-layer bytes & FLOPs ------------------------------------
d, dff = tc["hidden_size"], tc["intermediate_size"]
attn_p = 2*d*d                       # q_proj (16*128) + o_proj
attn_p += 2*(tc["num_key_value_heads"]*tc["head_dim"])*d  # k,v
attn_p += d*tc["num_attention_heads"] # gate_proj
conv_p = 3*d*d + d*d + d*7           # in_proj + out_proj + depthwise
ffn_p = 3*d*dff
print("\n## Analytic per-layer weights (text)")
print(f"conv layer   : {conv_p:,} params ({conv_p*2/2**20:.2f} MiB bf16)")
print(f"attn layer   : {attn_p:,} params ({attn_p*2/2**20:.2f} MiB bf16)  [q,o full; k,v GQA 4:1; gate]")
print(f"ffn (SwiGLU) : {ffn_p:,} params ({ffn_p*2/2**20:.2f} MiB bf16)")

# arithmetic intensity per token (batch=1 decode step): FLOPs / weight bytes
def ai(flops, byts): return flops / byts
print("\n## Decode arithmetic intensity (per token, W=bf16, weights-only streaming)")
print(f"FFN  w1/w3 : {ai(2*2*d*dff, 2*2*d*dff)/1:.1f} FLOP/B")
print(f"FFN  w2    : {ai(2*dff*d, 2*dff*d):.1f} FLOP/B")
print(f"conv in    : {ai(2*3*d*d, 2*3*d*d):.1f} FLOP/B")
print(f"conv out   : {ai(2*d*d, 2*d*d):.1f} FLOP/B")
print(f"depthwise  : {ai(2*7*d, 2*7*d):.1f} FLOP/B  (elementwise, memory bound)")

# ---- verify against header -----------------------------------------------
got = {r[0] for r in rows}
exp = set()
for li, lt in enumerate(tc["layer_types"]):
    if lt == "conv":
        for s in ("in_proj.weight", "out_proj.weight", "conv.weight"):
            exp.add(f"model.language_model.layers.{li}.conv.{s}")
print("\n## checks")
print("conv layer tensors present:", all(e in got for e in exp))
print("tied embeddings (no lm_head):", "lm_head.weight" not in got)
print("sample conv names:", [n for n in sorted(got) if ".layers.0.conv." in n])
print("sample attn names:", [n for n in sorted(got) if ".layers.2.self_attn." in n][:12])

with open(os.path.join(os.path.dirname(__file__), "..", "results", "arch_tensors.tsv"), "w") as f:
    f.write("name\tshape\tdtype\tnumel\tbytes\n")
    for name, shape, dt, numel, nb in sorted(rows):
        f.write(f"{name}\t{shape}\t{dt}\t{numel}\t{nb}\n")
print("\nwrote results/arch_tensors.tsv")
