import torch,sys
sys.path.insert(0,'scripts')
from phase1_baseline import *
from transformers import AutoModelForImageTextToText, AutoProcessor
M=MODEL
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
m=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={"":0},trust_remote_code=True).eval()
txt="The quick brown fox jumps over the lazy dog. "*200
r=ppl(m,proc.tokenizer,txt,window=128,stride=64)
print("ppl",r)
# brute force single-window check
ids=proc.tokenizer(txt,return_tensors="pt").input_ids.to('cuda')
with torch.inference_mode():
    lg=m(input_ids=ids[:,:128],use_cache=False).logits.float()
l=-torch.log_softmax(lg[0,:-1],-1).gather(-1,ids[0,1:128,None]).squeeze(-1)
print("brute ppl first window", float(torch.exp(l.mean())), "n",l.numel())
# dynamic cache decode
from transformers.cache_utils import DynamicCache
try:
    c=DynamicCache(config=m.config.text_config); print("cache ok",c)
except Exception as e: print("cache err",e)
print(decode_timing(m, ids[:,:64],4))
# video
clips,med=video_items(); print(len(clips),"clips median",med,[round(c['motion'],4) for c in clips])
r=ask_video(m,proc,clips[0],64); print("vid",r['pred'],r['n_in'],r['seconds'],r['text'][:120].replace('\n',' '))
r=ask_image(m,proc,FRAMES+"/V1_static_surveillance_10.png",IMG_MC[0][1],64); print("img",r['pred'],r['n_in'],r['seconds'],r['text'][:200].replace('\n',' '))
