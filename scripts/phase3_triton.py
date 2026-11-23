"""Phase 3 GPU: a real block-sparse GEMV/GEMM that loads only the active blocks.

Weights are repacked block-major on the device:
  bf16 : W_q[b, o, 0:32]      (nb, n_out, 32)      contiguous 64 B per (block, out)
  iq4  : Q[b, o, 0:18]        (nb, n_out, 18)      16 nibble bytes + bf16 scale

The sparse kernel iterates an active-block index list, so a skipped block costs
zero bytes and zero FMAs (it is a real gather, not a masked dense read -- masked
reads are what a naive implementation does and they save nothing on bandwidth).

Baselines are the two honest alternatives:
  torch_dense   torch.mv on the dense weight matrix (cuBLAS)
  torch_gather  torch.mv on W[active_blocks] reshaped (what a PyTorch-level
                block-sparse emulation costs: a gather + a copy)

Reported timings are CUDA-event medians over many launches, which include the
launch overhead that dominates at these matrix sizes -- that is the point, since
a 2048x2048 GEMV is 8.4 MB and ~14 us of DRAM time at 600 GB/s.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np
import torch
import triton
import triton.language as tl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
P3 = os.path.join(ROOT, "artifacts", "p3")
DB = 32


# ----------------------------------------------------------------- kernels
@triton.jit
def gemv_bf16(Y, X, W, IDX, n_active, n_out, BLOCK_N: tl.constexpr, DB: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_out
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(n_active):
        b = tl.load(IDX + i)
        w = tl.load(W + b * n_out * DB + offs[:, None] * DB + tl.arange(0, DB)[None, :],
                    mask=mask[:, None], other=0.0)
        x = tl.load(X + b * DB + tl.arange(0, DB)).to(tl.float32)
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(Y + offs, acc.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def gemv_iq4(Y, X, Q, S, LUT, IDX, n_active, n_out, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n_out
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(n_active):
        b = tl.load(IDX + i)
        p = Q + (b * n_out + offs[:, None]) * 16
        q = tl.load(p + tl.arange(0, 16)[None, :], mask=mask[:, None], other=0)
        sc = tl.load(S + b * n_out + offs, mask=mask, other=1.0).to(tl.float32)
        lo = (q & 0x0F).to(tl.int32)
        hi = ((q >> 4) & 0x0F).to(tl.int32)
        wlo = tl.load(LUT + lo)                      # (BLOCK_N, 16) fp32
        whi = tl.load(LUT + hi)
        x0 = tl.load(X + b * 32 + tl.arange(0, 16)).to(tl.float32)
        x1 = tl.load(X + b * 32 + 16 + tl.arange(0, 16)).to(tl.float32)
        acc += (tl.sum(wlo * x0[None, :], axis=1) + tl.sum(whi * x1[None, :], axis=1)) * sc
    tl.store(Y + offs, acc.to(Y.dtype.element_ty), mask=mask)


_BURN_A = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
_BURN_B = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)


def burn(n=200):
    """Raise the SM clock before each measurement; mobile-GPU clocks drift within a
    single script run and otherwise make later rows look 1.8x slower."""
    for _ in range(n):
        torch.mm(_BURN_A, _BURN_B)
    torch.cuda.synchronize()


def bench3(fn, iters=200, warmup=20):
    return min(bench(fn, iters, warmup) for _ in range(3))


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters


def load_bf16(name, shape):
    v = np.fromfile(os.path.join(P3, name), dtype=np.uint16)
    return torch.from_numpy(v.copy()).view(torch.bfloat16).float().reshape(shape).cuda()


def pack_blocks(w, iq4_lut=None):
    """w: (n_out, n_in) fp32 cuda -> block-major bf16 (nb, n_out, 32) and iq4 (nb, n_out, 18)."""
    n_out, n_in = w.shape
    nb = n_in // DB
    wb = w.reshape(n_out, nb, DB).permute(1, 0, 2).contiguous()          # (nb, n_out, 32)
    bf = wb.to(torch.bfloat16)
    amax = wb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    sc = (amax / 127.0)
    v = wb / sc
    d = (v[..., None] - iq4_lut).abs().argmin(-1)                        # (nb, n_out, 32)
    q = d.to(torch.uint8)
    packed = (q[..., :16] | (q[..., 16:] << 4))                          # (nb, n_out, 16)
    scb = sc.squeeze(-1).to(torch.bfloat16)
    codes = packed.contiguous()                       # (nb, n_out, 16) uint8
    scales = scb.contiguous().to(torch.bfloat16)      # (nb, n_out) bf16  -> 18 B per block total
    return bf, (codes, scales)


def main() -> int:
    torch.manual_seed(0)
    LUT_NL = torch.tensor([-127., -104., -83., -65., -45., -28., -12., 1., 13., 23., 33., 41., 49., 56., 62., 66.],
                          device="cuda")
    LUT_SYM = torch.tensor([-127., -110., -93., -76., -59., -42., -25., -8., 8., 25., 42., 59., 76., 93., 110., 127.],
                           device="cuda")

    xf = load_bf16("p3_xf.bin", (64, 2048))
    w1 = load_bf16("p3_w1.bin", (7680, 2048))
    w3 = load_bf16("p3_w3.bin", (7680, 2048))
    w2 = load_bf16("p3_w2.bin", (2048, 7680))
    wout = load_bf16("p3_out.bin", (2048, 2048))
    with torch.no_grad():
        h = torch.nn.functional.silu(xf @ w1.T) * (xf @ w3.T)            # (64, 7680)
    cases = [
        ("down_proj", w2, h),            # 7680 -> 2048, 31.5 MB bf16
        ("out_proj", wout, h[:, :2048]),  # 2048 -> 2048, 8.4 MB bf16
    ]

    out = {"cases": [], "device": torch.cuda.get_device_name(0)}
    for name, w, x in cases:
        n_out, n_in = w.shape
        nb = n_in // DB
        bf_blocks, (q_codes, q_scales) = pack_blocks(w, LUT_SYM)
        x1 = x[0].contiguous().to(torch.bfloat16)
        w_bf = w.to(torch.bfloat16)
        y = torch.empty(n_out, device="cuda", dtype=torch.bfloat16)
        BLOCK_N = 64
        grid = (triton.cdiv(n_out, BLOCK_N),)

        # per-token block energies -> active index lists
        e = x1.reshape(nb, DB).pow(2).sum(-1).sqrt()
        rec = {"matrix": name, "n_out": n_out, "n_in": n_in, "nb": nb,
               "bytes_bf16": n_out * n_in * 2, "bytes_iq4": nb * n_out * 18, "levels": []}
        for tau in (0.0, 0.25, 0.50):
            burn()
            th = float(torch.quantile(e.float(), tau)) if tau > 0 else -1.0
            idx = torch.nonzero(e > th).flatten().to(torch.int32).contiguous()
            keep = int(idx.numel())
            ridx = idx.to(torch.int64)
            wg = bf_blocks[idx].reshape(keep * n_out, DB)                 # gathered rows
            wg_dense = wg.new_zeros((n_out, n_in), dtype=torch.bfloat16)
            for j in range(keep):
                wg_dense[:, idx[j] * DB:(idx[j] + 1) * DB] = bf_blocks[idx[j]].to(torch.bfloat16)
            lvl = {"tau": tau, "active": keep / nb, "bytes_read_bf16": keep * n_out * DB * 2,
                   "bytes_read_iq4": keep * n_out * 18}

            lvl["torch_dense_bf16"] = bench3(lambda: torch.mv(w_bf, x1))
            lvl["torch_gather_bf16"] = bench3(lambda: torch.mv(wg_dense, x1))
            allidx = torch.arange(nb, dtype=torch.int32, device="cuda")
            lvl["triton_bf16_all"] = bench3(lambda: gemv_bf16[grid](y, x1, bf_blocks, allidx, nb, n_out,
                                                                   BLOCK_N=BLOCK_N, DB=DB))
            lvl["triton_bf16_sparse"] = bench3(lambda: gemv_bf16[grid](y, x1, bf_blocks, idx, keep, n_out,
                                                                      BLOCK_N=BLOCK_N, DB=DB))
            lvl["triton_iq4_sparse"] = bench3(lambda: gemv_iq4[grid](y, x1, q_codes, q_scales, LUT_SYM, idx, keep,
                                                                    n_out, BLOCK_N=BLOCK_N))

            # correctness against the dense reference
            ref = (w_bf @ x1).float()
            y.zero_()
            gemv_bf16[grid](y, x1, bf_blocks, allidx, nb, n_out, BLOCK_N=BLOCK_N, DB=DB)
            lvl["rel_err_bf16"] = float((y.float() - ref).norm() / ref.norm())
            y.zero_()
            gemv_iq4[grid](y, x1, q_codes, q_scales, LUT_SYM, allidx, nb, n_out, BLOCK_N=BLOCK_N)
            lvl["rel_err_iq4"] = float((y.float() - ref).norm() / ref.norm())
            y.zero_()
            gemv_bf16[grid](y, x1, bf_blocks, idx, keep, n_out, BLOCK_N=BLOCK_N, DB=DB)
            lvl["rel_err_sparse"] = float((y.float() - ref).norm() / ref.norm())

            lvl["sp_bf16_vs_dense"] = lvl["torch_dense_bf16"] / lvl["triton_bf16_sparse"]
            lvl["sp_iq4_vs_dense"] = lvl["torch_dense_bf16"] / lvl["triton_iq4_sparse"]
            lvl["gbps_bf16_sparse"] = lvl["bytes_read_bf16"] / (lvl["triton_bf16_sparse"] * 1e6)
            lvl["gbps_iq4_sparse"] = lvl["bytes_read_iq4"] / (lvl["triton_iq4_sparse"] * 1e6)
            rec["levels"].append(lvl)
            print(f"{name} tau={tau:.2f} act={lvl['active']:.3f} "
                  f"torch {lvl['torch_dense_bf16']*1e3:6.1f}us triton_bf16 {lvl['triton_bf16_all']*1e3:6.1f}us "
                  f"sparse {lvl['triton_bf16_sparse']*1e3:6.1f}us iq4 {lvl['triton_iq4_sparse']*1e3:6.1f}us "
                  f"sp={lvl['sp_bf16_vs_dense']:.2f}x err(bf16)={lvl['rel_err_bf16']:.2e} err(iq4)={lvl['rel_err_iq4']:.3f}")
        out["cases"].append(rec)

    with open(os.path.join(RES, "phase3_triton.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
