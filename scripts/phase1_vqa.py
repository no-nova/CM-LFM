#!/usr/bin/env python3
"""Phase 1 quality baseline: position-balanced image MC-VQA + pairwise temporal
motion discrimination + cut detection. All ground truth is either human-verified
per image (scene identity) or *measured from pixels* (motion energy, cut ratio),
so nothing is annotated by a model.

Design notes
------------
* Every MC item is asked in BOTH option orders (correct answer at A and at B),
  so a constant-letter policy scores 50%, not 100%.
* Pairwise motion: "which clip moves more" with the pair order flipped; ground
  truth = mean |frame_t - frame_{t-1}| computed from the actual pixels.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_baseline import MODEL, FRAMES, RES, MemProbe, cuda_time  # noqa

LETTER = re.compile(r"\b([AB])\b")

# (file, correct option text, three distractors, tag)
IMG_ITEMS = [
    ("V1_static_surveillance_10.png", "a narrow indoor hallway with people",
     ["a wide sandy beach", "a mountain forest", "a crowded stadium"], "hallway"),
    ("V2_local_motion_10.png", "a highway with traffic under a cloudy sky",
     ["an underground subway station", "a snowy village", "a swimming pool"], "highway"),
    ("V3_fast_natural_object_10.png", "a woman in front of an abstract painting",
     ["a dog running on grass", "a bowl of fruit on a table", "a city skyline at night"], "newsreader"),
    ("V4_camera_pan_10.png", "a small boat moving on water near a rocky shore",
     ["a train inside a tunnel", "a desert with camels", "a library full of books"], "coastguard"),
    ("V5_illumination_change_10.png", "a large cargo ship loaded with containers",
     ["a bicycle race", "an empty office", "a waterfall"], "container_ship"),
    ("V6_multi_shot_cuts_10.png", "two news anchors seated at a desk",
     ["two chefs cooking", "two soccer players", "two musicians on stage"], "anchors"),
]
EXTRA = os.path.join(RES, "p1_frames")
IMG_ITEMS += [
    (os.path.join(EXTRA, "bus.png"), "a white bus on a street with traffic",
     ["an airplane at an airport", "a sailboat at sea", "a farm with sheep"], "bus"),
    (os.path.join(EXTRA, "flower.png"), "a field of colorful flowers with a windmill",
     ["a snowy mountain peak", "a busy factory floor", "a dark cave"], "flower_field"),
    (os.path.join(EXTRA, "husky.png"), "people running with dogs by a river",
     ["swimmers in a pool", "a wedding ceremony indoors", "cars parked in a garage"], "runners_dogs"),
    (os.path.join(EXTRA, "tempete.png"), "yellow flowers and ferns beside a wooden log fence",
     ["a beach at sunset", "a rocket launch", "an empty highway"], "flowers_fence"),
]


def load_clips(n_frames=8):
    from PIL import Image
    clips = []
    for sc in sorted({f.rsplit("_", 1)[0] for f in os.listdir(FRAMES)}):
        files = sorted(f for f in os.listdir(FRAMES) if f.startswith(sc + "_"))
        for start in (1, 19):
            sel = files[start:start + n_frames]
            arrs = [np.asarray(Image.open(os.path.join(FRAMES, f)).convert("RGB")) for f in sel]
            d = np.array([np.abs(arrs[i].astype(np.int16) - arrs[i - 1].astype(np.int16)).mean() / 255.0
                          for i in range(1, len(arrs))])
            clips.append({"clip": f"{sc}_{start:02d}", "scenario": sc, "motion": float(d.mean()),
                          "diffs": d.tolist(), "frames": np.stack(arrs),
                          "cut": bool(d.max() > 0.05)})   # 0.05 = 12.75/255 mean abs diff, measured
    return clips


@torch.inference_mode()
def gen(model, proc, msgs, max_new, videos_kwargs=None):
    kw = {"processor_kwargs": {"videos_kwargs": videos_kwargs}} if videos_kwargs else {}
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                     return_dict=True, return_tensors="pt", **kw).to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.time()
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    dt = time.time() - t0
    txt = proc.tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
    body = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S)
    return {"text": txt[-500:], "body": body[-500:], "n_in": int(n_in),
            "n_out": int(out.shape[1] - n_in), "seconds": round(dt, 2)}


def pick(resp, pat):
    m = pat.findall(resp["body"].upper())
    return m[-1] if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=320)
    a = ap.parse_args()
    probe = MemProbe()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()

    # ---------------- image MC, both orders ----------------
    img_rows = []
    rng = np.random.default_rng(0)
    for path, correct, distract, tag in IMG_ITEMS:
        if not os.path.isabs(path):
            path = os.path.join(FRAMES, path)
        img = Image.open(path).convert("RGB")
        for order in (0, 1):
            opts = [correct] + list(distract)
            # order 0: correct at A; order 1: correct at B (first distractor takes A)
            if order == 0:
                seq, gt = opts, "A"
            else:
                seq, gt = [opts[1], opts[0], opts[2], opts[3]], "B"
            q = ("Which best describes this image?\n"
                 + "\n".join(f"({l}) {t}" for l, t in zip("ABCD", seq))
                 + "\nAnswer with a single letter (A/B/C/D).")
            r = gen(model, proc, [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": q}]}], a.max_new)
            pred = pick(r, re.compile(r"\b([ABCD])\b"))
            img_rows.append({"item": tag, "order": order, "gt": gt, "pred": pred,
                             "correct": pred == gt, **{k: r[k] for k in ("n_in", "n_out", "seconds", "text")}})
            print(f"[img] {tag:14s} order={order} gt={gt} pred={pred} n_out={r['n_out']} {r['seconds']}s")

    # ---------------- temporal: pairwise motion + cut detection ----------------
    clips = load_clips()
    clips.sort(key=lambda c: c["motion"])
    low, high = clips[:6], clips[6:]
    pairs = list(zip(high, low))
    vid_rows = []
    for i, (hi, lo) in enumerate(pairs):
        for order in (0, 1):  # which clip is shown first; GT letter = the more-moving one
            first, second = (hi, lo) if order == 0 else (lo, hi)
            gt = "A" if order == 0 else "B"
            q = ("Two video clips follow. Which one contains more motion/change over time?\n"
                 "Answer with a single letter: A (first clip) or B (second clip).")
            try:
                r = gen(model, proc, [{"role": "user", "content": [
                    {"type": "video", "video": first["frames"]},
                    {"type": "video", "video": second["frames"]},
                    {"type": "text", "text": q}]}], a.max_new, {"fps": 30.0})
                pred = pick(r, re.compile(r"\b([AB])\b"))
            except Exception as ex:
                r = {"text": f"ERROR {ex}", "body": "", "n_in": 0, "n_out": 0, "seconds": 0}
                pred = "?"
            vid_rows.append({"task": "pair_motion", "pair": f"{hi['clip']}|{lo['clip']}",
                             "motion_hi": round(hi["motion"], 4), "motion_lo": round(lo["motion"], 4),
                             "order": order, "gt": gt, "pred": pred, "correct": pred == gt,
                             **{k: r[k] for k in ("n_in", "n_out", "seconds", "text")}})
            print(f"[pair] {hi['clip'][:28]:28s} vs {lo['clip'][:28]:28s} order={order} gt={gt} pred={pred}")
    # cut detection on 6 clips (3 with measured cuts if available)
    for c in clips:
        q = ("Does this video contain an abrupt scene cut (a hard change between shots)? "
             "Answer with one word: yes or no.")
        try:
            r = gen(model, proc, [{"role": "user", "content": [
                {"type": "video", "video": c["frames"]}, {"type": "text", "text": q}]}],
                a.max_new, {"fps": 30.0})
            b = r["body"].lower()
            pred = "yes" if re.search(r"\byes\b", b) else ("no" if re.search(r"\bno\b", b) else "?")
        except Exception as ex:
            r = {"text": f"ERROR {ex}", "body": "", "n_in": 0, "n_out": 0, "seconds": 0}
            pred = "?"
        gt = "yes" if c["cut"] else "no"
        vid_rows.append({"task": "cut", "clip": c["clip"], "motion": round(c["motion"], 4),
                         "max_diff": round(max(c["diffs"]), 4), "gt": gt, "pred": pred,
                         "correct": pred == gt, **{k: r[k] for k in ("n_in", "n_out", "seconds", "text")}})
        print(f"[cut ] {c['clip'][:32]:32s} maxd={max(c['diffs']):.4f} gt={gt} pred={pred}")

    # frozen-clip control: GT is exact (0.0 measured diff), 3 frozen + 3 moving
    for c in (clips[0], clips[5], clips[9]):
        frozen = np.repeat(c["frames"][:1], len(c["frames"]), axis=0)
        q = ("Does this video contain any movement or change over time? "
             "Answer exactly one word: yes or no.")
        try:
            r = gen(model, proc, [{"role": "user", "content": [
                {"type": "video", "video": frozen}, {"type": "text", "text": q}]}],
                a.max_new, {"fps": 30.0})
            b = r["body"].lower()
            pred = "yes" if re.search(r"\byes\b", b) else ("no" if re.search(r"\bno\b", b) else "?")
        except Exception as ex:
            r = {"text": f"ERROR {ex}", "body": "", "n_in": 0, "n_out": 0, "seconds": 0}
            pred = "?"
        vid_rows.append({"task": "frozen_control", "clip": c["clip"] + "_frozen", "motion": 0.0,
                         "gt": "no", "pred": pred, "correct": pred == "no",
                         **{k: r[k] for k in ("n_in", "n_out", "seconds", "text")}})
        print(f"[froz] {c['clip'][:32]:32s} gt=no pred={pred}")

    def acc(rows):
        return round(float(np.mean([r["correct"] for r in rows])), 4) if rows else None
    out = {"image": img_rows, "video": vid_rows,
           "image_acc": acc(img_rows), "video_pair_acc": acc([v for v in vid_rows if v["task"] == "pair_motion"]),
           "video_cut_acc": acc([v for v in vid_rows if v["task"] == "cut"]),
           "video_frozen_acc": acc([v for v in vid_rows if v["task"] == "frozen_control"]),
           "image_acc_order0": acc([r for r in img_rows if r["order"] == 0]),
           "image_acc_order1": acc([r for r in img_rows if r["order"] == 1]),
           "latency": {"image_s": round(float(np.mean([r["seconds"] for r in img_rows])), 2),
                       "video_s": round(float(np.mean([r["seconds"] for r in vid_rows])), 2)},
           "mem_peak_mb": probe.close()}
    json.dump(out, open(os.path.join(RES, "phase1_vqa.json"), "w"), indent=1)
    print("[vqa]", {k: out[k] for k in ("image_acc", "image_acc_order0", "image_acc_order1",
                                       "video_pair_acc", "video_cut_acc", "video_frozen_acc",
                                       "latency", "mem_peak_mb")})


if __name__ == "__main__":
    main()
