#!/usr/bin/env python3
"""Phase 1 CPU baselines for LFM3-VL-2B (Zen 5, AVX-512).

Two independent measurements, both under the 1.5 GB anonymous-RSS budget:

A. Whole-model CPU forward in bf16 with ZERO-COPY mmap weights: the text
   backbone is instantiated on the meta device and then `assign`-ed tensors that
   are views into the safetensors mmap, so weights live in page cache (RssFile),
   never in anonymous memory. Gives real end-to-end CPU prefill/decode latency
   and a CPU-vs-GPU numerical reference at the identical bf16 precision.

B. Operator-level microbenchmarks with the exact shapes of this architecture
   (d=2048, d_ff=7680, k=7 depthwise conv, 16 heads x 128, 4 KV heads), bf16 and
   fp32, 1 and 32 threads, OMP_WAIT_POLICY=passive. Reports attained GB/s,
   GFLOPS and the fraction of the Zen 5 socket's ~208 GB/s DRAM roof.

Usage: OMP_WAIT_POLICY=passive python scripts/phase1_cpu.py [--threads 1,32]
"""
import argparse, gc, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
RES = os.path.join(ROOT, "results")
D, DFF, K, HEADS, KV, HD = 2048, 7680, 7, 16, 4, 128
# Measured on this host with scripts/bw_probe.cpp (results/phase1_bw.txt): 32-thread AVX-512
# read 43.99 GB/s, copy 17.65 GB/s (8.8 per direction), triad 20.85 GB/s. Effectively one
# DDR5 channel of usable bandwidth, so the GEMV roof is the *read* figure, not a 2-channel spec.
DRAM_READ_GBS = 44.0
DRAM_TRIAD_GBS = 20.9


def rss():
    s = open("/proc/self/status").read()
    g = lambda k: int(s.split(k)[1].split()[0]) // 1024
    return {"anon": g("RssAnon:"), "file": g("RssFile:"), "rss": g("VmRSS:")}


def bench(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def op_table(threads):
    torch.set_num_threads(threads)
    rows = []
    cases = []

    def add(name, flops, bytes_, fn):
        cases.append((name, flops, bytes_, fn))

    for dt, bpe in ((torch.bfloat16, 2), (torch.float32, 4)):
        tag = "bf16" if dt == torch.bfloat16 else "fp32"
        for T in (1, 16, 64):
            # attention projections + FFN GEMMs, exact shapes from the anatomy
            for label, out_f, in_f in (("q_proj/o_proj", D, D), ("k_proj/v_proj", KV * HD, D),
                                       ("ffn.w1/w3", DFF, D), ("ffn.w2", D, DFF),
                                       ("conv.in_proj", 3 * D, D), ("conv.out_proj", D, D)):
                w = torch.randn(out_f, in_f, dtype=dt)
                x = torch.randn(T, in_f, dtype=dt)
                add(f"mm_{tag}_T{T}_{label}_{out_f}x{in_f}", 2 * out_f * in_f * T,
                    out_f * in_f * bpe + T * in_f * bpe, lambda w=w, x=x: x @ w.T)
        # depthwise causal short conv, k=7, groups=d
        for T in (1, 16, 64, 256):
            w = torch.randn(D, 1, K, dtype=dt)
            x = torch.randn(1, D, T + K - 1, dtype=dt)
            add(f"dwconv_{tag}_T{T}_k{K}_g{D}", 2 * D * K * T, (D * K + D * (T + K - 1)) * bpe,
                lambda w=w, x=x: torch.nn.functional.conv1d(x, w, groups=D))
        for T in (16, 64):
            x = torch.randn(T, D, dtype=dt)
            g = torch.randn(T, D, dtype=dt)
            add(f"swiglu_{tag}_T{T}", 4 * T * D, 3 * T * D * bpe,
                lambda x=x, g=g: torch.nn.functional.silu(x) * g)
            w = torch.randn(D, dtype=dt)
            add(f"rmsnorm_{tag}_T{T}", 3 * T * D, 2 * T * D * bpe,
                lambda x=x, w=w: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(dt) * w)
    # the two LFM3Conv elementwise gates (B*x and C*h) are two more T*D multiplies
    for name, flops, bytes_, fn in cases:
        ms = bench(fn)
        rows.append({"op": name, "threads": threads, "median_ms": round(ms, 4),
                     "gflops": round(flops / ms / 1e6, 2),
                     "gbs": round(bytes_ / ms / 1e6, 1),
                     "pct_of_read_roof": round(100.0 * bytes_ / ms / 1e6 / DRAM_READ_GBS, 1)})
    return rows


def mmap_model():
    """Instantiate on meta, then assign mmap-backed CPU tensors (zero copy)."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
    st = {}
    with safe_open(os.path.join(MODEL, "model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            st[k] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(st, strict=False, assign=True)
    del st
    gc.collect()
    model.tie_weights()
    # non-persistent buffers (RotaryEmbedding inv_freq) are not in the checkpoint and
    # stay on the meta device -> rebuild them from the config
    fixed = []
    for name, m in model.named_modules():
        if type(m).__name__.endswith("RotaryEmbedding") and any(b.is_meta for b in m.buffers()):
            cfgm = cfg.vision_config if ".vision" in name else cfg.text_config
            type(m).__init__(m, cfgm)
            fixed.append(name)
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta:
        raise RuntimeError(f"still meta: {meta[:8]}")
    print(f"[mmap] rebuilt rotary buffers: {fixed}; unexpected keys: {list(unexpected)[:4]}")
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,32")
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    out = {"env": {"OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"),
                   "torch": torch.__version__, "mkl": torch.backends.mkl.is_available()},
           "rss_after_import": rss()}

    if not a.skip_model:
        t0 = time.time()
        model = mmap_model()
        out["mmap_load_seconds"] = round(time.time() - t0, 2)
        out["rss_after_mmap_load"] = rss()
        lm = model.model.language_model
        # cross-device numerical reference: same bf16 weights, CPU vs GPU kernels
        torch.manual_seed(0)
        ids = torch.randint(0, 1000, (1, 64))
        with torch.inference_mode():
            t_cpu = time.perf_counter()
            ref = lm(input_ids=ids, use_cache=False).last_hidden_state.float()
            out["cpu_prefill_64_ms"] = round((time.perf_counter() - t_cpu) * 1e3, 1)
            out["rss_after_forward"] = rss()
            gpu = None
            if torch.cuda.is_available():
                g = mmap_model_gpu()
                with torch.inference_mode():
                    gpu = g.model.language_model(input_ids=ids.cuda(), use_cache=False).last_hidden_state.float().cpu()
                del g
                torch.cuda.empty_cache()
        if gpu is not None:
            d = (ref - gpu).abs()
            out["cpu_vs_gpu"] = {"hidden_rmse": float(d.pow(2).mean().sqrt()), "max_abs": float(d.max()),
                                 "rel_l2": float(d.norm() / gpu.norm()), "seq_len": 64,
                                 "note": "identical bf16 weights; CPU AVX-512 bf16 vs GPU tensor-core bf16"}
        del ref, gpu
        gc.collect()
        # CPU decode with cache
        from transformers.cache_utils import DynamicCache
        if torch.cuda.is_available():
            pass
        cache = DynamicCache(config=model.config.text_config)
        with torch.inference_mode():
            lm(input_ids=ids[:, :8], past_key_values=cache, use_cache=True)
            ts = []
            for _ in range(6):
                t = time.perf_counter()
                o = lm(input_ids=ids[:, :1], past_key_values=cache, use_cache=True)
                ts.append((time.perf_counter() - t) * 1e3)
        out["cpu_decode_ms_per_token"] = round(sorted(ts)[len(ts) // 2], 2)
        out["rss_after_decode"] = rss()
        del model, lm, cache, o
        gc.collect()

    out["ops"] = []
    for th in [int(t) for t in a.threads.split(",")]:
        out["ops"] += op_table(th)
    out["rss_end"] = rss()
    json.dump(out, open(os.path.join(RES, "phase1_cpu.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "ops"}, indent=1))
    for r in out["ops"]:
        print(f"{r['op']:44s} th={r['threads']:2d} {r['median_ms']:9.4f} ms "
              f"{r['gflops']:8.1f} GF  {r['gbs']:7.1f} GB/s  {r['pct_of_read_roof']:5.1f}% of 44 GB/s")


def mmap_model_gpu():
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()


if __name__ == "__main__":
    main()
