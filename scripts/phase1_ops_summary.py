import csv, json, re, collections
rows=list(csv.DictReader(open('results/phase1_ops.csv')))
tot=sum(float(r['self_cuda_us']) for r in rows)
def cat(op):
    o=op.lower()
    if 'gemm' in o or o=='aten::mm' or 'aten::addmm' in o or 'aten::linear' in o: return 'GEMM (BF16 tensor core, mm/addmm)'
    if 'attention_forward' in o or 'fmha' in o or 'aten::scaled_dot_product' in o: return 'attention (FMHA)'
    if 'conv' in o: return 'depthwise conv1d'
    if 'pow' in o: return 'rsqrt/pow (norm)'
    if 'copy' in o: return 'memory copy / layout'
    if 'mul' in o: return 'elementwise mul (gates, silu*mul)'
    if 'silu' in o or 'sigmoid' in o or 'exp' in o or 'tanh' in o: return 'elementwise silu/sigmoid'
    if 'reduce' in o or 'sum' in o or 'mean' in o: return 'reductions (norm/loss)'
    if 'embed' in o: return 'embedding'
    if 'softmax' in o: return 'softmax'
    return 'other ('+op[:40]+')'
agg=collections.Counter(); cnt=collections.Counter()
for r in rows:
    agg[cat(r['op'])]+=float(r['self_cuda_us']); cnt[cat(r['op'])]+=int(r['count'])
print(f"total self CUDA time: {tot/1000:.2f} ms over {len(rows)} distinct kernels\n")
for k,v in agg.most_common():
    print(f"{v/1000:8.2f} ms  {100*v/tot:5.1f}%  n={cnt[k]:4d}  {k}")
json.dump({k:{"ms":round(v/1000,3),"pct":round(100*v/tot,2),"calls":cnt[k]} for k,v in agg.most_common()},
          open('results/phase1_ops_summary.json','w'), indent=1)
