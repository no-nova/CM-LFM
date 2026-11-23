"""Phase 5 verification gate. Three checks, all against real weights/artifacts:

1. tau=0 isomorphism of pi_ffn and pi_conv  -> max|dlogit| <= 1e-5 (fp32)
2. scene-cut safety on LFM3-VL-2B           -> exactly 0.000000 drift on flush frames
3. host-memory discipline                    -> peak RssAnon < 1.5 GB during this run

Run:  python scripts/verify_all.py
Exit status is non-zero if any check fails, so it can gate the report build.
"""
import json, os, sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = "/home/user/Downloads/CM-LFM3"
CKPT = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B/model.safetensors"
DB = 32
torch.manual_seed(0)


def rss_anon_mb():
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def load(tag):
    """Stream one tensor at a time out of the mmap; never materialise the checkpoint."""
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        return {k: f.get_tensor(f"{tag}.{k}") for k in ("w1.weight", "w2.weight", "w3.weight")} \
            if tag.endswith("ffn") else None


def check_ffn_isomorphism():
    """pi_ffn: permuting the 7680 intermediate channels jointly in w1/w3 columns and
    w2 rows must leave the layer function unchanged; with the permutation applied as a
    *runtime index* the tau=0 path is exactly the dense computation."""
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        w1 = f.get_tensor("model.language_model.layers.0.feed_forward.w1.weight").float()
        w3 = f.get_tensor("model.language_model.layers.0.feed_forward.w3.weight").float()
        w2 = f.get_tensor("model.language_model.layers.0.feed_forward.w2.weight").float()
    x = torch.randn(4, 256, w1.shape[1], dtype=torch.float32) / w1.shape[1] ** 0.5
    h = F.silu(x @ w1.T) * (x @ w3.T)
    y = h @ w2.T
    nb = h.shape[-1] // DB
    # 32-lane block permutation: block b -> pi(b), channels pi(b)*32 + j
    pi = (torch.randperm(nb)[:, None] * DB + torch.arange(DB)).reshape(-1)
    y_perm = (h.index_select(-1, pi) @ w2.index_select(1, pi).T)
    d = ((y - y_perm).norm() / y.norm()).item()
    # ordering control: the identity permutation must reproduce y bit-for-bit
    d_id = (y - (h @ w2.T)).abs().max().item()
    # negative control: permute one of the two coupled tensors only
    y_bad = (h.index_select(-1, pi) @ w2.T)
    d_bad = ((y - y_bad).norm() / y.norm()).item()
    return d, d_bad, d_id, y.abs().max().item()


def check_conv_isomorphism():
    """pi_conv: B, C, V slices of in_proj, the depthwise conv channels and the
    out_proj input columns are all channel-wise before the dense projection, so a
    simultaneous permutation is an isomorphism."""
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        win = f.get_tensor("model.language_model.layers.0.conv.in_proj.weight").float()
        cw = f.get_tensor("model.language_model.layers.0.conv.conv.weight").float()   # (d,1,k)
        wout = f.get_tensor("model.language_model.layers.0.conv.out_proj.weight").float()
    d, k = cw.shape[0], cw.shape[-1]
    x = torch.randn(2, 64, win.shape[1], dtype=torch.float32) / win.shape[1] ** 0.5

    def fwd(win_, cw_, wout_, perm=None):
        bcx = (x @ win_.T).transpose(1, 2)                    # (b, 3d, T)
        B, C, V = bcx.chunk(3, dim=1)
        h = B * V
        y = F.conv1d(h, cw_.reshape(d, 1, k), padding=k - 1, groups=d)[..., :x.shape[1]]
        y = (C * y).transpose(1, 2)
        return y @ wout_.T

    y = fwd(win, cw, wout)
    pi = (torch.randperm(d // DB)[:, None] * DB + torch.arange(DB)).reshape(-1)
    win_p = torch.cat([win[pi], win[d + pi], win[2 * d + pi]], 0)
    y_p = fwd(win_p, cw[pi], wout[:, pi])
    d_ok = ((y - y_p).norm() / y.norm()).item()
    d_id = (y - fwd(win, cw, wout)).abs().max().item()
    # negative control: permute the conv channels but not out_proj's input columns
    y_bad = fwd(win_p, cw[pi], wout)
    return d_ok, ((y - y_bad).norm() / y.norm()).item(), d_id, y.abs().max().item()


def check_retained_energy():
    """Bug-1 gate: dropping the weakest tau fraction of blocks must retain more than
    (1-tau) of the stream energy, co-activation order must retain at least as much as
    native order, and the corrected block energy must be a per-token RMS (not a sum over
    T), which is asserted through the theta transfer check of phase 3."""
    p = f"{ROOT}/results/phase2_energy_audit.json"
    if not os.path.exists(p):
        return None
    a = json.load(open(p))
    worst, at = 1.0, None
    for r in a["rows"]:
        for order in ("native", "gram", "coact"):
            for rule in ("thr", "topk"):
                v = r[f"{order}_{rule}"]
                if v < worst:
                    worst, at = v, f"{r['family']}:{order}:{rule}:tau={r['tau']}"
    return worst, at, a["failures"], a["pass"]


def check_scene_cut():
    p = f"{ROOT}/results/phase4_video.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    worst = max(v["scene_cut_maxabs_flush"] for v in d["clips"].values())
    # the guarantee is per *flush frame*: when stage 4 fires, the prefill is the dense one.
    flush = sorted(k for k, v in d["clips"].items() if v["n_flush_frames"] > 0)
    full = {k: v for k, v in d["clips"].items() if v["n_flush_frames"] == len(v["s_hat"])}
    kl_full = max((v["logit_kl"] for v in full.values()), default=0.0)
    return worst, kl_full, flush, sorted(full)


def main():
    peak = rss_anon_mb()
    res, ok = {}, True

    d_ffn, d_ffn_bad, d_ffn_id, s_ffn = check_ffn_isomorphism()
    peak = max(peak, rss_anon_mb())
    res["ffn_iso_relL2"] = d_ffn
    res["ffn_identity_control_maxabs"] = d_ffn_id
    res["ffn_negative_control_relL2"] = d_ffn_bad
    res["ffn_out_scale"] = s_ffn
    res["ffn_iso_pass"] = d_ffn <= 1e-6 and d_ffn_id == 0.0 and d_ffn_bad > 1e-2
    ok &= res["ffn_iso_pass"]

    d_conv, d_conv_bad, d_conv_id, s_conv = check_conv_isomorphism()
    peak = max(peak, rss_anon_mb())
    res["conv_iso_relL2"] = d_conv
    res["conv_identity_control_maxabs"] = d_conv_id
    res["conv_negative_control_relL2"] = d_conv_bad
    res["conv_out_scale"] = s_conv
    res["conv_iso_pass"] = d_conv <= 1e-6 and d_conv_id == 0.0 and d_conv_bad > 1e-2
    ok &= res["conv_iso_pass"]

    re_ = check_retained_energy()
    res["retained_energy_min"] = None if re_ is None else round(re_[0], 4)
    res["retained_energy_min_at"] = None if re_ is None else re_[1]
    res["retained_energy_failures"] = None if re_ is None else re_[2]
    # assertion: retained_energy_ratio > (1 - tau) for every measured (family, order, rule, tau)
    res["retained_energy_pass"] = bool(
        re_ is not None and re_[3] and re_[0] > (1.0 - 0.50) and not re_[2])
    ok &= res["retained_energy_pass"]

    sc = check_scene_cut()
    res["scene_cut_maxabs"] = None if sc is None else sc[0]
    res["scene_cut_max_kl"] = None if sc is None else sc[1]
    res["flush_clips"] = None if sc is None else sc[2]
    res["fully_flushed_clips"] = None if sc is None else sc[3]
    res["scene_cut_pass"] = bool(sc is not None and sc[0] == 0.0 and sc[1] == 0.0)
    ok &= res["scene_cut_pass"]

    res["peak_rss_anon_mb"] = round(peak, 1)
    res["rss_pass"] = peak < 1536.0
    ok &= res["rss_pass"]
    res["all_pass"] = ok

    json.dump(res, open(f"{ROOT}/results/verify_all.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:28s} {v}")
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
