import threading,time,torch,os
peak={"anon":0,"file":0,"rss":0}
stop=False
def mon():
    while not stop:
        try:
            s=open('/proc/self/status').read()
            g=lambda k: int(s.split(k)[1].split()[0])//1024
            peak["anon"]=max(peak["anon"],g('RssAnon:')); peak["file"]=max(peak["file"],g('RssFile:')); peak["rss"]=max(peak["rss"],g('VmRSS:'))
        except Exception: pass
        time.sleep(0.01)
th=threading.Thread(target=mon,daemon=True); th.start()
from transformers import AutoModelForImageTextToText
M="/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
m=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={"":0},trust_remote_code=True)
print("after load",peak)
x=torch.randint(0,1000,(1,1024),device='cuda')
with torch.no_grad(): y=m(input_ids=x,use_cache=False) if False else m.model.language_model(input_ids=x,use_cache=False)
print("after fwd",peak,"cuda",round(torch.cuda.max_memory_allocated()/2**30,2))
stop=True
