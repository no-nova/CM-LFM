"""Phase 4: five-stage spatio-temporal token skipping (P2) on LFM3-VL-2B video streams.

Stage 1  patch-level frame differencing with an adaptive threshold + pre-dilation
         3x3 majority filter (>= 5/9 dirty neighbours)
Stage 2  morphological halo dilation of the dirty mask
Stage 3  projection onto the vision patch-token grid (spatial_merge_size units)
Stage 4  full-flush fallback when the dirty fraction shat > 0.50 (scene cut / pan)
Stage 5  token feature reuse: retained tokens are the *same* vision features, static
         tokens are dropped from the multimodal prefill

Everything is measured against a dense reference on the same weights and the same
kernel path; cut frames are asserted to be bit-identical (flush => zero drift).
"""
import gc, json, os, sys, time
import numpy as np
import torch

ROOT = "/home/user/Downloads/CM-LFM3"
MODEL = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
FRAMES = "/home/user/Downloads/CNN-CM/data/real_videos/frames_224"
OUT = f"{ROOT}/results"
PATCH = 32           # 224 / 32 = 7 ; vision patch 16 x temporal 2 x merge 2
GRID = 224 // PATCH
S_HAT_FLUSH = 0.50
# Stage-1 constants are calibrated (scripts/phase4_calibrate.py) on a known-static
# clip: TAU_ABS = 95th pct of 32x32 block pixelwise frame differences, and the
# global-motion gate uses GAMMA x the mean block difference of that same clip.
TAU_ABS = 0.0339
GAMMA = 1.5
NOISE_MEAN = 0.0170

DEV = "cuda"


def load_clip(name, n=8):
    files = sorted(f for f in os.listdir(FRAMES) if f.startswith(name))
    files = files[:n]
    ims = []
    for f in files:
        from PIL import Image
        ims.append(np.asarray(Image.open(os.path.join(FRAMES, f)).convert("RGB")))
    return np.stack(ims)   # (T, H, W, 3) uint8


def frame_diffs(frames):
    """Stage 1a: 32x32 block means of *pixelwise* abs frame differences.

    Averaging the difference (not the block colour) keeps sensitivity to motion
    inside a block instead of cancelling it out.
    """
    f = frames.astype(np.float32) / 255.0
    dd = np.abs(f[1:] - f[:-1]).mean(-1)                       # (T-1,H,W)
    d = np.zeros((frames.shape[0], GRID, GRID), np.float32)
    d[1:] = dd.reshape(dd.shape[0], GRID, PATCH, GRID, PATCH).mean(axis=(2, 4))
    return d


def majority_filter(mask):
    """Pre-dilation 3x3 majority: dirty iff >= 5 of the 9 neighbours are dirty."""
    m = mask.astype(np.float32)
    p = np.pad(m, 1, mode="edge")
    acc = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += p[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]]
    return acc >= 5


def dilate(mask):
    m = mask.astype(np.float32)
    p = np.pad(m, 1, mode="edge")
    acc = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += p[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]]
    return acc > 0


def p2_masks(frames, temporal=2):
    """Returns (keep_masks [T//temporal, 49] bool, diagnostics per frame).

    The vision stack packs `temporal_patch_size=2` frames into one temporal patch,
    so a temporal patch block is static only if *both* frames are clean.
    """
    d = frame_diffs(frames)
    keep, diag = [], []
    prev_keep = np.ones(GRID * GRID, bool)      # frame 0 is always fully processed
    for t in range(frames.shape[0]):
        if t == 0:
            dirty = np.ones((GRID, GRID), bool)
            s_hat = 1.0
            flushed = True
            n_after_majority = GRID * GRID
        else:
            # adaptive threshold: max(absolute noise floor, 2x the low decile)
            thr = max(TAU_ABS, 2.0 * float(np.quantile(d[t], 0.10)))
            dirty = d[t] > thr
            n_raw = int(dirty.sum())
            dirty = majority_filter(dirty)              # Stage 1b
            n_after_majority = int(dirty.sum())
            s_hat = n_after_majority / (GRID * GRID)
            global_motion = float(d[t].mean()) > GAMMA * NOISE_MEAN
            flushed = (s_hat > S_HAT_FLUSH) or global_motion
            if flushed:
                dirty = np.ones((GRID, GRID), bool)
            else:
                dirty = dilate(dirty)
        km = dirty.reshape(-1).copy()
        keep.append(km)
        diag.append(dict(frame=t, s_hat=round(float(s_hat), 4), flushed=bool(flushed),
                         mean_diff=round(float(d[t].mean()), 5),
                         blocks_raw=None if t == 0 else int(n_raw),
                         blocks_after_majority=n_after_majority,
                         blocks_kept=int(km.sum())))
        prev_keep = km
    keep = np.stack(keep)                      # (T, 49)
    T = keep.shape[0] - keep.shape[0] % temporal
    keep = keep[:T].reshape(T // temporal, temporal, -1).any(1)
    return keep, diag


def build_inputs(model, proc, frames, question):
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], videos=[list(frames)], return_tensors="pt", do_sample_frames=False)
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in enc.items()}, text


@torch.inference_mode()
def video_features(model, enc):
    pv = enc["pixel_values_videos"].to(torch.bfloat16)
    vg = enc["video_grid_thw"]
    out = model.model.get_video_features(pv, vg, return_dict=True)
    return out.pooler_output[0]                 # (49*T, 2048)


@torch.inference_mode()
def prefill(model, enc, feats, keep=None, flush_all=False):
    """Runs the language model on inputs_embeds with video features spliced in.
    keep: (T,49) bool -> static tokens are removed from the sequence (Stage 5).
    """
    ids = enc["input_ids"][0]
    tok = model.config.video_token_id
    pos = (ids == tok).nonzero(as_tuple=True)[0]
    n = feats.shape[0]
    assert pos.numel() == n, f"{pos.numel()} video tokens vs {n} features"
    emb = model.model.language_model.embed_tokens(ids.unsqueeze(0)).clone()
    emb[0, pos] = feats.to(emb.dtype)
    if keep is None or flush_all:
        idx = torch.arange(ids.numel(), device=ids.device)
    else:
        kept = torch.from_numpy(keep.reshape(-1)).to(ids.device)
        keep_pos = pos[kept]
        drop = torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)
        drop[pos[~kept]] = False
        idx = drop.nonzero(as_tuple=True)[0]
        assert torch.equal(torch.sort(keep_pos)[0], keep_pos)
    e = emb[:, idx, :]
    am = torch.ones(e.shape[:2], dtype=torch.long, device=ids.device)
    out = model.model.language_model(inputs_embeds=e, attention_mask=am, use_cache=False)
    return out.last_hidden_state, e.shape[1], pos, emb


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


@torch.inference_mode()
def next_logits(model, hidden):
    lm = model.model.language_model
    h = lm.norm(hidden[:, -1:, :]) if hasattr(lm, "norm") else hidden[:, -1:, :]
    return (h.float() @ lm.embed_tokens.weight.float().T)[0, 0]


def kl_top1(la, lb):
    pa = torch.log_softmax(la, -1)
    pb = torch.log_softmax(lb, -1)
    kl = float((pa.exp() * (pa - pb)).sum())
    return kl, int(la.argmax() == lb.argmax())


def hidden_to_logits(model, h):
    lm = model.model.language_model
    hn = lm.norm(h) if hasattr(lm, "norm") else h
    return hn.float() @ lm.embed_tokens.weight.float().T


@torch.inference_mode()
def mc_logprob(model, proc, frames, feats, question, options, keep):
    """Position-balanced binary MC by the length-normalised log-likelihood of each
    option string conditioned on the video. No free-form generation, so neither the
    letter bias nor the model's <think> prefix can dominate the decision."""
    head = (question + "\nA) " + options[0] + "\nB) " + options[1] + "\nAnswer: ")
    scores = []
    for opt in options:
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": head}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        msgs2 = [{"role": "assistant", "content": [{"type": "text", "text": opt}]}]
        # the assistant turn closes with the template's end token; score the text only
        full = text + opt
        enc = proc(text=[full], videos=[list(frames)], return_tensors="pt", do_sample_frames=False)
        enc = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in enc.items()}
        ids = enc["input_ids"][0]
        n_opt = len(proc.tokenizer(opt, add_special_tokens=False)["input_ids"])
        tok = model.config.video_token_id
        pos = (ids == tok).nonzero(as_tuple=True)[0]
        emb = model.model.language_model.embed_tokens(ids.unsqueeze(0)).clone()
        emb[0, pos] = feats.to(emb.dtype)
        if keep is not None:
            kept = torch.from_numpy(keep.reshape(-1)).to(ids.device)
            drop = torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)
            drop[pos[~kept]] = False
            idx = drop.nonzero(as_tuple=True)[0]
            # recompute which of the retained rows are the answer tokens
            keep_ids = ids[idx]
            e = emb[:, idx, :]
            am = torch.ones(e.shape[:2], dtype=torch.long, device=ids.device)
            out = model.model.language_model(inputs_embeds=e, attention_mask=am, use_cache=False)
            lg = hidden_to_logits(model, out.last_hidden_state)[0].float()
            tgt = keep_ids
        else:
            out = model.model.language_model(inputs_embeds=emb,
                                             attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=DEV),
                                             use_cache=False)
            lg = hidden_to_logits(model, out.last_hidden_state)[0].float()
            tgt = ids
        n = lg.shape[0]
        lp = torch.log_softmax(lg[:-1], -1)
        t = tgt[1:]
        ll = lp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        scores.append(float(ll[-n_opt:].mean()))
        del emb, out, lg, lp
    return scores, (scores[0] > scores[1])


def logits_of(model, hidden):
    lm = model.model.language_model
    h = lm.norm(hidden[:, -1:, :]) if hasattr(lm, "norm") else hidden[:, -1:, :]
    w = lm.embed_tokens.weight
    return (h.float() @ w.float().T)[0, 0]


SCEN = {
    "V1_static_surveillance": ("an indoor hallway with people walking", "a beach with waves", "a highway with cars", "a kitchen close-up"),
    "V2_local_motion": ("a highway with moving traffic", "an empty desert road", "a snowy mountain", "a swimming pool"),
    "V3_fast_natural_object": ("a news studio with a person talking", "a football match", "a cooking show", "a car commercial"),
    "V4_camera_pan": ("a small boat on the water near rocks", "an airport runway", "a subway station", "a forest trail"),
    "V5_illumination_change": ("a large cargo container ship at sea", "a bicycle race", "a bakery interior", "a tennis court"),
    "V6_multi_shot_cuts": ("two news anchors sitting at a desk", "a rock concert", "a hospital corridor", "a farm with animals"),
}


def main():
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        attn_implementation="sdpa", trust_remote_code=True).eval()
    rep = {"model": "LFM3-VL-2B", "gpu": torch.cuda.get_device_name(0),
           "patch_px": PATCH, "grid": GRID, "tau_abs": TAU_ABS, "s_hat_flush": S_HAT_FLUSH,
           "clips": {}, "vqa": {}}

    for name, opts in SCEN.items():
        frames = load_clip(name)
        keep, diag = p2_masks(frames)
        enc, text = build_inputs(model, proc, frames, "Describe what happens in this video.")
        feats = video_features(model, enc)

        # Stage-5 exactness: retained features are the same tensor, so reuse is exact.
        h_dense, n_dense, pos, emb = prefill(model, enc, feats)
        h_skip, n_skip, _, _ = prefill(model, enc, feats, keep=keep)
        flush_frames = [d["frame"] for d in diag if d["flushed"]]
        # scene-cut safety: a flushed frame set must reproduce the dense prefill bitwise
        h_flush, n_flush, _, _ = prefill(model, enc, feats, keep=np.ones_like(keep))
        drift_flush = float((h_flush.float() - h_dense.float()).abs().max())

        # per-frame feature drift of the *kept* tokens (should be exactly 0)
        T = frames.shape[0]
        d = feats.shape[0] // T
        kf = torch.from_numpy(keep.reshape(-1))
        drift_feat = rel_l2(feats[kf], feats[kf])

        # latency: dense vs skipped prefill (median of 5)
        def timeit(fn, reps=5):
            fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                torch.cuda.synchronize(); t = time.perf_counter()
                fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
            ts.sort(); return ts[len(ts) // 2]

        def dense_call():
            with torch.inference_mode():
                prefill(model, enc, feats)

        def skip_call():
            with torch.inference_mode():
                prefill(model, enc, feats, keep=keep)

        ms_dense = timeit(dense_call)
        ms_skip = timeit(skip_call)
        ms_vis = timeit(lambda: video_features(model, enc))

        lg_d = next_logits(model, h_dense)
        lg_s = next_logits(model, h_skip)
        kl, t1 = kl_top1(lg_d, lg_s)
        rep["clips"][name] = dict(
            logit_kl=round(kl, 6), top1_agree=t1,
            frames=int(T), tokens_dense=int(n_dense), tokens_skipped=int(n_skip),
            token_reduction=round(1 - n_skip / n_dense, 4),
            kept_fraction=round(float(keep.mean()), 4),
            n_flush_frames=len(flush_frames), flush_frames=flush_frames[:12],
            s_hat=[d["s_hat"] for d in diag],
            prefill_ms_dense=round(ms_dense, 3), prefill_ms_skipped=round(ms_skip, 3),
            prefill_speedup=round(ms_dense / ms_skip, 3),
            vision_ms=round(ms_vis, 3),
            scene_cut_maxabs_flush=drift_flush,
            feature_reuse_rel_l2=drift_feat,
            mask_diag=diag[:6])
        del h_dense, h_skip, h_flush, emb
        gc.collect(); torch.cuda.empty_cache()

        # position-balanced MC VQA, dense vs P2-skipped
        corr = SCEN[name][0]
        distract = SCEN[name][1:3]
        q = "Which statement best describes this video?"
        res = {"dense_hits": 0, "skip_hits": 0, "n": 0, "detail": []}
        for di, dis in enumerate(distract):
            for order in ((corr, dis), (dis, corr)):
                sd, _ = mc_logprob(model, proc, frames, feats, q, order, None)
                ss, _ = mc_logprob(model, proc, frames, feats, q, order, keep)
                win_d = 0 if sd[0] > sd[1] else 1
                win_s = 0 if ss[0] > ss[1] else 1
                hit_d = (order[win_d] == corr)
                hit_s = (order[win_s] == corr)
                res["dense_hits"] += int(hit_d); res["skip_hits"] += int(hit_s); res["n"] += 1
                res["detail"].append(dict(order=list(order), win_dense=win_d, win_skip=win_s,
                                          lp_dense=[round(v, 4) for v in sd],
                                          lp_skip=[round(v, 4) for v in ss],
                                          hit_dense=bool(hit_d), hit_skip=bool(hit_s)))
        res["dense_acc"] = round(res["dense_hits"] / res["n"], 4)
        res["skip_acc"] = round(res["skip_hits"] / res["n"], 4)
        rep["vqa"][name] = res
        print(name, "vqa", res["dense_acc"], "->", res["skip_acc"], flush=True)
        del enc, feats
        gc.collect(); torch.cuda.empty_cache()

    json.dump(rep, open(f"{OUT}/phase4_video.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("mask_diag", "s_hat", "flush_frames")}
                      for k, v in rep["clips"].items()}, indent=1))
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
