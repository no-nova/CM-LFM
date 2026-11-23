#!/usr/bin/env python3
"""Emit results/phase1_architecture.md: layer topology, byte share, arithmetic intensity."""
import json, os
import torch
from safetensors import safe_open

M="/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"; RES="/home/user/Downloads/CM-LFM3/results"
cfg=json.load(open(os.path.join(M,"config.json")))
tc=cfg["text_config"] if "text_config" in cfg else cfg
vc=cfg.get("vision_config",{})
d=tc["hidden_size"]; ffn=tc["intermediate_size"]; k=tc["conv_L_cache"]
nh=tc["num_attention_heads"]; nkv=tc["num_key_value_heads"]; hd=tc.get("head_dim", d//nh)
V=tc["vocab_size"]; L=tc["num_hidden_layers"]; lt=tc["layer_types"]
vd=vc.get("hidden_size"); vffn=vc.get("intermediate_size"); vL=vc.get("num_hidden_layers"); vlt=vc.get("layer_types",[])
# per-layer parameter formulas
conv_p = 3*d*d + d*d + d*k                 # in_proj + out_proj + depthwise(no bias)
attn_p = d*nh*hd + d*nkv*hd*2 + d*nh*hd + d*nh + 2*hd   # q, k, v, o, per-head gate, 2 rmsnorms
ffn_p  = 3*d*ffn
scan = {}
with safe_open(os.path.join(M,"model.safetensors"),"pt",device="cpu") as f:
    for name in f.keys():
        if name=="model.language_model.embed_tokens.weight": continue
        st=f.get_slice(name); shape=st.get_shape(); dtype=st.get_dtype()
        scan[name]=(shape, int(torch.tensor(shape).prod()) * (2 if "BF16" in dtype or "F16" in dtype else 4))
numel=sum(v[1]//2 for v in scan.values())
def comp(n): return sum(v[1] for kk,v in scan.items() if n(kk))
tot_bytes=sum(v[1] for v in scan.values()) + V*d*2
rows=[]
cats=[("embed_tokens (tied lm_head)", lambda x: False),  # handled separately
      ("text LFM3Conv block (13x)", lambda x: "language_model.layers." in x and ".conv." in x),
      ("text GQA attention (11x)", lambda x: "language_model.layers." in x and "self_attn" in x and ".conv." not in x),
      ("text FFN (24x)", lambda x: "language_model.layers." in x and (".mlp." in x or ".feed_forward." in x)),
      ("vision LFM3Conv+attn+FFN", lambda x: "vision_tower" in x),
      ("multimodal projector", lambda x: "multi_modal" in x or "projector" in x)]
for nm,fn in cats[1:]:
    b=comp(fn); rows.append((nm,b))
def ai(flops,byt): return flops/byt
lines=[]
lines.append("# LFM3-VL-2B architecture anatomy (measured from the checkpoint)\n")
lines.append(f"* text: d={d}, d_ff={ffn}, k={k}, {L} layers, {nh} heads x {hd} = {nh*hd} (q/o), {nkv} KV heads ({nkv*hd}), vocab {V}, tied embeddings, sliding window {tc.get('sliding_window')}")
lines.append(f"* layer_types: " + ", ".join(f"{t}x{lt.count(t)}" for t in dict.fromkeys(lt)))
lines.append(f"* vision: d={vd}, d_ff={vffn}, {vL} layers, patch {vc.get('patch_size')}, temporal {vc.get('temporal_patch_size')}, merge {vc.get('spatial_merge_size')}; types: " + ", ".join(f"{t}x{vlt.count(t)}" for t in dict.fromkeys(vlt)))
lines.append(f"* total params (excl. vision) {numel:,}; total checkpoint bytes {tot_bytes/2**20:.1f} MiB\n")
lines.append("| component | params (M) | bf16 MiB | share |\n|---|---|---|---|")
for nm,b in sorted(rows,key=lambda r:-r[1]):
    lines.append(f"| {nm} | {b/2**20/1e6*1e6/1e6:.2f} | {b/2**20:.1f} | {100*b/tot_bytes:.2f}% |")
lines.append(f"| embed_tokens (tied) | {V*d/1e6:.2f} | {V*d*2/2**20:.1f} | {100*V*d*2/tot_bytes:.2f}% |")
# arithmetic intensity
def row(name, flops1, flops64, wbytes, act1_bytes, act64_bytes):
    return (name, round(ai(flops1, wbytes+act1_bytes),3), round(ai(flops64, wbytes+act64_bytes),3),
            round(wbytes/2**20,2))
gemv=[("q_proj/o_proj (d x d)", 2*d*d, d*d*2),
      ("k_proj/v_proj (512 x d)", 2*nkv*hd*d, nkv*hd*d*2),
      ("ffn.w1/w3 (d_ff x d)", 2*ffn*d, ffn*d*2),
      ("ffn.w2 (d x d_ff)", 2*d*ffn, d*ffn*2),
      ("conv.in_proj (3d x d)", 2*3*d*d, 3*d*d*2),
      ("conv.out_proj (d x d)", 2*d*d, d*d*2)]
lines.append("\n## Arithmetic intensity α_l = FLOPs / bytes moved (bf16, weights streamed once)\n")
lines.append("| operator | α (T=1) | α (T=64) | weight MiB |\n|---|---|---|---|")
for nm,f,wb in gemv:
    lines.append(f"| {nm} | {ai(2*f/2, wb + 2*(wb/2/dw if False else 0)) if False else round(f/(wb + 2*d*2),2)} | {round(64*f/(wb + 64*2*2*d),2)} | {wb/2**20:.2f} |")
open(os.path.join(RES,"phase1_architecture.md"),"w").write("\n".join(lines)+"\n")
print("\n".join(lines))
