#!/usr/bin/env python3
"""Cold (DRAM-resident) CPU operator roofline for LFM3-VL-2B shapes.

The warm microbenchmarks in phase1_cpu.py point at L3 (a 31.5 MB bf16 FFN matrix
fits in the 64 MB L3), so they overstate achievable DRAM bandwidth. Here every GEMM
draws from a pool of 8 independent weight copies (>= 250 MB) cycled per repetition,
so each call streams cold data from DRAM. This is the number a decode step actually
sees: one touch per weight element per token.

Also measures the LFM3Conv "unfused tax": the reference path in modeling_lfm3.py
does in_proj -> transpose -> split -> gate -> conv1d -> gate -> transpose ->
contiguous -> out_proj, i.e. 3 extra full-size round trips through memory that a
fused kernel removes.
"""
import json, os, statistics, time
import torch

D, DFF, K, HEADS, KV, HD = 2048, 7680, 7, 16, 4, 128
ROOF = 44.0  # measured 32-thread AVX-512 read bandwidth, results/phase1_bw.txt
RES = "/home/user/Downloads/CM-LFM3/results"


def med(fn, rep=9, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def cold_gemv(threads):
    torch.set_num_threads(threads)
    rows = []
    shapes = [("q_proj 2048x2048", D, D), ("kv_proj 512x2048", KV * HD, D),
              ("w1/w3 7680x2048", DFF, D), ("w2 2048x7680", D, DFF),
              ("conv in_proj 6144x2048", 3 * D, D)]
    for dt, name_dt, bpe in ((torch.bfloat16, "bf16", 2), (torch.float32, "fp32", 4)):
        for label, o, i in shapes:
            pool = [torch.randn(o, i, dtype=dt) for _ in range(8)]
            x = torch.randn(1, i, dtype=dt)
            idx = [0]
            def run():
                idx[0] = (idx[0] + 1) % 8
                return x @ pool[idx[0]].T
            ms = med(run)
            nbytes = o * i * bpe
            rows.append({"op": f"cold_gemv_{name_dt}_{label}", "threads": threads,
                         "median_ms": round(ms, 4), "weight_mb": round(nbytes / 2**20, 1),
                         "gbs": round(nbytes / ms / 1e6, 1),
                         "pct_of_44GBs": round(100 * nbytes / ms / 1e6 / ROOF, 1),
                         "gflops": round(2 * o * i / ms / 1e6, 1)})
            del pool
    return rows


def conv_tax(threads):
    """Unfused reference path vs a transposed-layout variant, T = 1 and 16."""
    torch.set_num_threads(threads)
    rows = []
    for dt, nm in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        for T in (1, 16, 64):
            w_in = torch.randn(3 * D, D, dtype=dt)
            w_out = torch.randn(D, D, dtype=dt)
            wc = torch.randn(D, 1, K, dtype=dt)
            x = torch.randn(T, D, dtype=dt)
            def ref():  # exactly modeling_lfm3.py Lfm3ShortConv.forward
                bcx = (x @ w_in.T).transpose(-1, -2)          # (d, 3T)
                b, c, h = bcx.chunk(3, dim=-2)
                h = b * h
                y = torch.nn.functional.conv1d(h, wc, groups=D, padding=K - 1)[..., :T]
                y = (c * y).transpose(-1, -2).contiguous()
                return y @ w_out.T
            def no_transpose():  # skip both transposes + contiguous, conv in (T, d)
                g = x @ w_in.T                                 # (T, 3d)
                b, c, h = g.split(D, dim=-1)
                h = (b * h).transpose(0, 1).unsqueeze(0)       # (1, d, T)
                y = torch.nn.functional.conv1d(h, wc, groups=D, padding=K - 1)[..., :T]
                return ((c.transpose(0, 1).unsqueeze(0) * y).squeeze(0).transpose(0, 1)) @ w_out.T
            ms_ref = med(ref)
            ms_nt = med(no_transpose)
            rows.append({"op": f"lfm3conv_{nm}_T{T}", "threads": threads,
                         "ref_ms": round(ms_ref, 4), "notranspose_ms": round(ms_nt, 4),
                         "transpose_tax_ms": round(ms_ref - ms_nt, 4),
                         "tax_pct": round(100 * (ms_ref - ms_nt) / ms_ref, 1)})
    return rows


def main():
    out = {"roof_gbs": ROOF, "cold_gemv": [], "conv_tax": []}
    for th in (1, 16):
        out["cold_gemv"] += cold_gemv(th)
        out["conv_tax"] += conv_tax(th)
    json.dump(out, open(os.path.join(RES, "phase1_cpu_cold.json"), "w"), indent=1)
    for r in out["cold_gemv"]:
        print(f"{r['op']:34s} th={r['threads']:2d} {r['median_ms']:8.4f} ms "
              f"{r['weight_mb']:6.1f} MB {r['gbs']:7.1f} GB/s {r['pct_of_44GBs']:5.1f}%")
    for r in out["conv_tax"]:
        print(f"{r['op']:26s} th={r['threads']:2d} ref={r['ref_ms']:8.4f} "
              f"notranspose={r['notranspose_ms']:8.4f} tax={r['tax_pct']:5.1f}%")


if __name__ == "__main__":
    main()
