#!/usr/bin/env python3
"""Phase 1 dense baseline for LFM3-VL-2B.

Memory contract (CLAUDE.md §1.2): single model, bf16 on GPU via low_cpu_mem_usage,
no torch.load of the checkpoint, peak RssAnon < 1.5 GB (checkpoint pages are
file-backed mmap and reclaimable; both are logged).

Tasks
  ppl       WikiText-2 test perplexity (sliding window)            -> results/phase1_ppl.json
  latency   prefill T in {16,64,256,1024}, decode 64 steps, VL path-> results/phase1_latency.json
  numeric   sdpa vs eager dense logits: RMSE / max|d| / top-1      -> results/phase1_numeric.json
  profile   torch.profiler op table for 1024-token prefill         -> results/phase1_ops.csv
  vqa       image MC (10 items) + video motion MC (12 clips)       -> results/phase1_vqa.json
"""
import argparse, gc, json, os, re, sys, threading, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
FRAMES = "/home/user/Downloads/CNN-CM/data/real_videos/frames_224"
RES = os.path.join(ROOT, "results")


# ---------------------------------------------------------------- memory probe
class MemProbe:
    def __init__(self):
        self.peak = {"anon": 0, "file": 0, "rss": 0}
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            try:
                s = open("/proc/self/status").read()
                g = lambda k: int(s.split(k)[1].split()[0]) // 1024
                self.peak["anon"] = max(self.peak["anon"], g("RssAnon:"))
                self.peak["file"] = max(self.peak["file"], g("RssFile:"))
                self.peak["rss"] = max(self.peak["rss"], g("VmRSS:"))
            except Exception:
                pass
            time.sleep(0.01)

    def close(self):
        self.stop = True
        self.t.join(timeout=1)
        return dict(self.peak)


# ---------------------------------------------------------------- data helpers
def get_wikitext2(path):
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    import urllib.request
    def fetch(url, tries=8):
        for i in range(tries):
            try:
                return json.load(urllib.request.urlopen(url, timeout=90))
            except Exception as e:
                print(f"  retry {i+1}/{tries}: {e}", flush=True)
                time.sleep(2 + 2 * i)
        raise RuntimeError("wikitext download failed")
    rows = []
    off = 0
    while True:
        url = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext"
               f"&config=wikitext-2-raw-v1&split=test&offset={off}&length=100")
        d = fetch(url)
        rows += [r["row"]["text"] for r in d["rows"]]
        off += 100
        print(f"  wikitext rows {off}/{d['num_rows_total']}", flush=True)
        if off >= d["num_rows_total"]:
            break
    txt = "\n".join(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(txt)
    return txt


@torch.inference_mode()
def ppl(model, tok, text, window=1024, stride=512):
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    n = ids.shape[1]
    nll = torch.zeros(n, dtype=torch.float64)
    seen = torch.zeros(n, dtype=torch.bool)
    t0 = time.time()
    for beg in range(0, n - window + 1, stride):
        end = beg + window
        out = model(input_ids=ids[:, beg:end], use_cache=False).logits.float()
        tgt = ids[0, beg + 1:end]
        l = -torch.log_softmax(out[0, :-1], -1).gather(
            -1, tgt[:, None]).squeeze(-1).double().cpu()   # l[j] predicts token beg+1+j
        if beg == 0:
            nll[1:end] = l
            seen[1:end] = True
        else:
            nll[end - stride:end] = l[window - stride - 1:]
            seen[end - stride:end] = True
    tot = nll[seen].sum()
    cnt = int(seen.sum())
    return {"ppl": float(torch.exp(tot / cnt)), "nll_sum": float(tot), "tokens": cnt,
            "bits_per_token": float(tot / cnt / np.log(2)), "tokens_total": int(n),
            "window": window, "stride": stride, "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- timing helpers
def cuda_time(fn, warmup=3, rep=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"median_ms": round(ts[len(ts) // 2], 3), "min_ms": round(ts[0], 3),
            "p90_ms": round(ts[int(0.9 * len(ts))], 3)}


@torch.inference_mode()
def decode_timing(model, ids, n_new=64):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config.text_config)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_new):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"decode_ms_per_token_median": round(ts[len(ts) // 2], 3),
            "decode_ms_per_token_min": round(ts[0], 3), "n_new": n_new,
            "tok_per_s": round(1000.0 / ts[len(ts) // 2], 2)}


# ---------------------------------------------------------------- VQA items
IMG_MC = [
    ("V1_static_surveillance_10.png",
     "Which best describes this image? (A) a narrow indoor hallway with people (B) a wide sandy beach (C) a mountain forest (D) a crowded stadium",
     "A", "indoor hallway"),
    ("V2_local_motion_10.png",
     "Which best describes this image? (A) a highway with traffic under a cloudy sky (B) an underground subway station (C) a snowy village (D) a swimming pool",
     "A", "highway"),
    ("V3_fast_natural_object_10.png",
     "Which best describes this image? (A) a woman in front of an abstract painting (B) a dog running on grass (C) a bowl of fruit on a table (D) a city skyline at night",
     "A", "newsreader"),
    ("V4_camera_pan_10.png",
     "Which best describes this image? (A) a small boat moving on water near a rocky shore (B) a train inside a tunnel (C) a desert with camels (D) a library full of books",
     "A", "coastguard boat"),
    ("V5_illumination_change_10.png",
     "Which best describes this image? (A) a large cargo ship loaded with containers (B) a bicycle race (C) an empty office (D) a waterfall",
     "A", "container ship"),
    ("V6_multi_shot_cuts_10.png",
     "Which best describes this image? (A) two news anchors seated at a desk (B) two chefs cooking (C) two soccer players (D) two musicians on stage",
     "A", "anchors"),
]
EXTRA_MC = [
    ("bus.png", "Which best describes this image? (A) a white bus on a street with traffic (B) an airplane at an airport (C) a sailboat at sea (D) a farm with sheep", "A", "bus"),
    ("flower.png", "Which best describes this image? (A) a field of colorful flowers with a windmill (B) a snowy mountain peak (C) a busy factory floor (D) a dark cave", "A", "flower field"),
    ("husky.png", "Which best describes this image? (A) people running with dogs by a river (B) swimmers in a pool (C) a wedding ceremony indoors (D) cars parked in a garage", "A", "husky/runners"),
    ("tempete.png", "Which best describes this image? (A) yellow flowers and ferns beside a wooden log fence (B) a beach at sunset (C) a rocket launch (D) an empty highway", "A", "flowers/fence"),
]
LETTER = re.compile(r"\b([ABCD])\b")


def ask_image(model, proc, path, question, max_new=384):
    from PIL import Image
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(path).convert("RGB")},
        {"type": "text", "text": question + " Answer with a single letter (A/B/C/D) and nothing else."}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    dt = time.time() - t0
    gen = out[:, n_in:]
    txt = proc.tokenizer.decode(gen[0], skip_special_tokens=True)
    # strip think block, take last standalone letter
    body = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S)
    letters = LETTER.findall(body.upper())
    pred = letters[-1] if letters else "?"
    return {"text": txt[:400], "pred": pred, "n_in": int(n_in),
            "n_out": int(gen.shape[1]), "seconds": round(dt, 2)}


def video_items():
    """12 clips (8 frames each) from the 6 scenario sets; GT motion label is *measured*
    from mean |frame_t - frame_{t-1}| so it is objective, not annotated."""
    from PIL import Image
    clips = []
    for sc in sorted({f.rsplit("_", 1)[0] for f in os.listdir(FRAMES)}):
        files = sorted(f for f in os.listdir(FRAMES) if f.startswith(sc + "_"))
        for k, start in enumerate([1, 19]):
            sel = files[start:start + 8]
            arrs = [np.asarray(Image.open(os.path.join(FRAMES, f)).convert("RGB"), dtype=np.float32) / 255.0 for f in sel]
            d = float(np.mean([np.abs(arrs[i] - arrs[i - 1]).mean() for i in range(1, len(arrs))]))
            clips.append({"clip": f"{sc}_{start:02d}", "files": sel, "motion": d,
                          "frames": np.stack(arrs)})
    med = float(np.median([c["motion"] for c in clips]))
    for c in clips:
        c["gt"] = "yes" if c["motion"] > med else "no"
    return clips, med


def ask_video(model, proc, clip, max_new=384):
    q = ("Does this video contain significant motion or change over time? "
         "Answer with one word: yes or no.")
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": clip["frames"]},
        {"type": "text", "text": q}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt",
                                      videos_kwargs={"fps": 30.0}).to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    dt = time.time() - t0
    txt = proc.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
    body = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S).lower()
    pred = "yes" if re.search(r"\byes\b", body) else ("no" if re.search(r"\bno\b", body) else "?")
    return {"text": txt[:400], "pred": pred, "n_in": int(n_in), "seconds": round(dt, 2)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="ppl,latency,numeric,profile,vqa")
    ap.add_argument("--max-new", type=int, default=384)
    a = ap.parse_args()
    tasks = set(a.tasks.split(","))
    os.makedirs(RES, exist_ok=True)
    probe = MemProbe()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    load_s = round(time.time() - t0, 2)
    lm = model.model.language_model
    summary = {"load_seconds": load_s, "vram_weights_gib": round(torch.cuda.memory_allocated() / 2**30, 3)}
    print(f"[load] {load_s}s vram={summary['vram_weights_gib']} GiB rss={probe.peak}")

    if "ppl" in tasks:
        txt = get_wikitext2(os.path.join(RES, "wikitext2_test.txt"))
        r = ppl(model, proc.tokenizer, txt)
        r["load_seconds"] = load_s
        summary["ppl"] = r
        json.dump(r, open(os.path.join(RES, "phase1_ppl.json"), "w"), indent=1)
        print("[ppl]", r)

    if "numeric" in tasks:
        ids = torch.randint(0, 1000, (1, 512), device="cuda")
        outs = {}
        for impl in ("sdpa", "eager"):
            model.config.text_config._attn_implementation = impl
            with torch.inference_mode():
                outs[impl] = model(input_ids=ids, use_cache=False).logits.float()
        d = (outs["sdpa"] - outs["eager"]).abs()
        agree = (outs["sdpa"].argmax(-1) == outs["eager"].argmax(-1)).float().mean().item()
        r = {"rmse": float(torch.sqrt(((outs["sdpa"] - outs["eager"]) ** 2).mean())),
             "max_abs": float(d.max()), "top1_agreement": agree, "seq_len": 512,
             "note": "same bf16 weights, different attention kernel path (bf16 SDPA vs eager matmul+softmax fp32)"}
        del outs, d
        gc.collect(); torch.cuda.empty_cache()
        summary["numeric"] = r
        json.dump(r, open(os.path.join(RES, "phase1_numeric.json"), "w"), indent=1)
        print("[numeric]", r)
        model.config.text_config._attn_implementation = "sdpa"

    if "latency" in tasks:
        gc.collect(); torch.cuda.empty_cache()
        res = {"by_T": {}}
        for T in (16, 64, 256, 1024):
            ids = torch.randint(0, 1000, (1, T), device="cuda")
            gc.collect(); torch.cuda.empty_cache()
            t = cuda_time(lambda: lm(input_ids=ids, use_cache=False), warmup=3, rep=10)
            t["tokens_per_s"] = round(T / (t["median_ms"] / 1000), 1)
            res["by_T"][T] = t
            print(f"[prefill T={T}] {t}")
            del ids
        ids = torch.randint(0, 1000, (1, 64), device="cuda")
        res["decode"] = decode_timing(model, ids, 64)
        print("[decode]", res["decode"])
        # VL path: one 256x256 image, measure prefill (vision + text)
        from PIL import Image
        img = Image.open(os.path.join(RES, "p1_frames", "bus.png")).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "Describe this image in one sentence."}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        t = cuda_time(lambda: model(**inputs, use_cache=False), warmup=2, rep=6)
        res["vl_prefill_256px"] = {**t, "n_in": int(inputs["input_ids"].shape[1]),
                                   "image_grid_thw": inputs.get("image_grid_thw").tolist(),
                                   "pixels": list(inputs["pixel_values"].shape)}
        print("[vl prefill]", res["vl_prefill_256px"])
        summary["latency"] = res
        json.dump(res, open(os.path.join(RES, "phase1_latency.json"), "w"), indent=1)

    if "profile" in tasks:
        import csv
        ids = torch.randint(0, 1000, (1, 1024), device="cuda")
        for _ in range(3):
            lm(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            lm(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            if ev.self_device_time_total > 0:
                rows.append((ev.key, round(ev.self_device_time_total, 1), ev.count))
        rows.sort(key=lambda r: -r[1])
        with open(os.path.join(RES, "phase1_ops.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["op", "self_cuda_us", "count"]); w.writerows(rows)
        print("[profile] top ops:", rows[:12])
        summary["profile_top"] = rows[:12]

    if "vqa" in tasks:
        items = [(os.path.join(FRAMES, f), q, g, tag) for f, q, g, tag in IMG_MC]
        items += [(os.path.join(RES, "p1_frames", f), q, g, tag) for f, q, g, tag in EXTRA_MC]
        vr = []
        for path, q, gt, tag in items:
            r = ask_image(model, proc, path, q, a.max_new)
            r.update({"item": tag, "gt": gt, "correct": r["pred"] == gt, "kind": "image"})
            vr.append(r)
            print(f"[img] {tag:20s} gt={gt} pred={r['pred']} nin={r['n_in']} {r['seconds']}s")
        out = {"image": vr, "image_acc": round(np.mean([v["correct"] for v in vr]), 4)}
        if "video" in tasks or True:
            clips, med = video_items()
            vv = []
            for c in clips:
                try:
                    r = ask_video(model, proc, c, a.max_new)
                except Exception as ex:  # noqa
                    r = {"text": f"ERROR {type(ex).__name__}: {ex}", "pred": "?", "seconds": 0, "n_in": 0}
                r.update({"clip": c["clip"], "motion": round(c["motion"], 5), "gt": c["gt"],
                          "correct": r["pred"] == c["gt"], "kind": "video"})
                vv.append(r)
                print(f"[vid] {c['clip']:32s} motion={c['motion']:.4f} gt={c['gt']:3s} pred={r['pred']:3s} nin={r['n_in']}")
            out["video"] = vv
            out["video_acc"] = round(np.mean([v["correct"] for v in vv]), 4)
            out["video_motion_median"] = med
        summary["vqa"] = {k: out[k] for k in ("image_acc", "video_acc", "video_motion_median") if k in out}
        json.dump(out, open(os.path.join(RES, "phase1_vqa.json"), "w"), indent=1)
        print("[vqa]", summary["vqa"])

    summary["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    summary["mem_peak_mb"] = probe.close()
    summary["tasks"] = sorted(tasks)
    json.dump(summary, open(os.path.join(RES, "phase1_summary.json"), "w"), indent=1)
    print("[done]", json.dumps({k: v for k, v in summary.items() if k != "profile_top"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
