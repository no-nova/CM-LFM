import torch,sys,re,json
sys.path.insert(0,'scripts')
from phase1_baseline import *
from transformers import AutoModelForImageTextToText, AutoProcessor
proc=AutoProcessor.from_pretrained(MODEL,trust_remote_code=True)
m=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={"":0},trust_remote_code=True).eval()
from PIL import Image
msgs=[{"role":"user","content":[{"type":"image","image":Image.open(FRAMES+"/V1_static_surveillance_10.png").convert("RGB")},{"type":"text","text":"Which best describes this image? (A) a narrow indoor hallway with people (B) a sandy beach (C) a forest (D) a stadium. Answer with a single letter."}]}]
inp=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(m.device)
print("n_in",inp["input_ids"].shape)
print("PROMPT:",repr(proc.tokenizer.decode(inp["input_ids"][0][-60:])))
with torch.inference_mode():
    out=m.generate(**inp,do_sample=False,max_new_tokens=200)
g=out[0,inp["input_ids"].shape[1]:]
print("GEN ids",g.tolist()[:40])
print("GEN text",repr(proc.tokenizer.decode(g,skip_special_tokens=False))[:600])
print("GEN clean",repr(proc.tokenizer.decode(g,skip_special_tokens=True))[:600])
