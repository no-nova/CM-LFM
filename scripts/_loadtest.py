import os, time, json, torch
from transformers import AutoConfig, AutoProcessor
M="/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
t=time.time()
cfg=AutoConfig.from_pretrained(M, trust_remote_code=True)
print("cfg ok", type(cfg).__name__, round(time.time()-t,2))
print("sub cfg:", type(cfg.text_config).__name__, type(cfg.vision_config).__name__)
tok=AutoProcessor.from_pretrained(M, trust_remote_code=True)
print("proc ok", type(tok).__name__, round(time.time()-t,2))
from transformers import AutoModelForImageTextToText
m=AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"":0}, trust_remote_code=True)
print("model ok", round(time.time()-t,2), "cuda alloc GiB", round(torch.cuda.memory_allocated()/2**30,2))
print("params", sum(p.numel() for p in m.parameters())/1e6, "M")
print("rss peak MB", round(int(open('/proc/self/status').read().split('VmHWM:')[1].split()[0])/1024,1))
