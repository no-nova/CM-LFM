"""Aggregate the revised Phase-3 kernel sweep into report tables + the end-to-end
decode projection.

Inputs  : results/phase3_kernels2_sym.csv   (block-major, corrected bf16 decode,
                                             per-token RMS block energy, low-quantile theta)
          results/phase3_triton.json        (RTX 5070 block-sparse GEMV)
Outputs : results/phase3_tables2.md
          results/tab_phase3_cpu2.tex       (T x threads x mode)
          results/tab_phase3_gpu.tex        (GPU block-sparse GEMV)
          results/phase3_e2e.json           (decode projection)
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# executed-weight byte fractions of the text decode path (from the checkpoint header)
FRAC = {"ffn": 0.5904, "conv": 0.1138, "gqa": 0.0603, "emb": 0.1367}
DECODE_MS_BF16 = 142.11          # measured, Phase 1 (CPU, bf16, 16 threads)
ROOF_MS = 3583.5 / 44.0
# bytes actually touched at T=1 after block-major IQ4_NL at tau=0.5 (from the line counts)
BYTES_AFTER_MB = 22.118 * 24 + 8.258 * 13 + 221.0 + 500.0          # streaming 3.58 GB at the 44 GB/s single-channel roof


def main() -> int:
    rows = list(csv.DictReader(open(os.path.join(RES, "phase3_kernels2_sym.csv"))))
    data: dict[tuple[str, int, int, str, float], dict] = {}
    for r in rows:
        data[(r["kind"], int(r["T"]), int(r["threads"]), r["mode"], float(r["tau"]))] = {
            "ms": float(r["ms"]), "gbps": float(r["gbps"]), "active": float(r["active"]),
            "lines": float(r["lines_MB"]), "rel": float(r["rel_l2"])}

    TAU = 0.50
    md = ["# Phase 3 (revised): block-major IQ4_NL with per-token RMS gating", ""]
    md.append("Microseconds per layer call, cold-flushed, sym codebook. Speedups are against "
              "dense bf16 at the same (kind, T, threads).")
    md.append("")
    md.append("| layer | T | th | bf16 | iq4bm | iq4bm+skip | iq4rm | iq4rm+skip | sp(bm+skip) | sp(rm+skip) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    tex = []
    for kind in ("ffn", "conv"):
        for th in (1, 16):
            for T in (1, 16, 64):
                g = lambda mode: data[(kind, T, th, mode, TAU)]["ms"] * 1e3
                b = g("bf16")
                sp_bm = b / g("iq4bm_skip")
                sp_rm = b / g("iq4rm_skip")
                md.append(f"| {kind} | {T} | {th} | {b:.1f} | {g('iq4bm'):.1f} | {g('iq4bm_skip'):.1f} | "
                          f"{g('iq4rm'):.1f} | {g('iq4rm_skip'):.1f} | {sp_bm:.2f}x | {sp_rm:.2f}x |")
                tex.append(f"{kind} & {T} & {th} & {g('bf16'):.1f} & {g('iq4bm'):.1f} & "
                           f"{g('iq4bm_skip'):.1f} & {g('iq4rm_skip'):.1f} & {sp_bm:.2f}$\\times$ & "
                           f"{sp_rm:.2f}$\\times$ \\\\")

    # cache-line accounting at T = 1, th = 16
    md += ["", "Cache lines touched (MB) at T=1, 16 threads, tau=0.50:", ""]
    md.append("| layer | bf16 | iq4 bm | iq4 bm+skip | iq4 rm+skip |")
    md.append("|---|---|---|---|---|")
    for kind in ("ffn", "conv"):
        md.append(f"| {kind} | {data[(kind,1,16,'bf16',TAU)]['lines']:.2f} | "
                  f"{data[(kind,1,16,'iq4bm',TAU)]['lines']:.2f} | "
                  f"{data[(kind,1,16,'iq4bm_skip',TAU)]['lines']:.2f} | "
                  f"{data[(kind,1,16,'iq4rm_skip',TAU)]['lines']:.2f} |")

    # end-to-end projection
    s_ffn = data[("ffn", 1, 16, "bf16", TAU)]["ms"] / data[("ffn", 1, 16, "iq4bm_skip", TAU)]["ms"]
    s_conv = data[("conv", 1, 16, "bf16", TAU)]["ms"] / data[("conv", 1, 16, "iq4bm_skip", TAU)]["ms"]
    tot = sum(FRAC.values())
    f = {k: v / tot for k, v in FRAC.items()}
    # bytes-proportional time model with measured per-op speedups
    t_units = f["ffn"] / s_ffn + f["conv"] / s_conv + f["gqa"] + f["emb"]
    proj_ms = DECODE_MS_BF16 * t_units
    e2e = {"s_ffn_T1": round(s_ffn, 3), "s_conv_T1": round(s_conv, 3),
           "frac": {k: round(v, 4) for k, v in f.items()},
           "time_units": round(t_units, 4), "speedup": round(1 / t_units, 3),
           "decode_ms_before": DECODE_MS_BF16, "decode_ms_projected": round(proj_ms, 2),
           "roof_ms": round(ROOF_MS, 2),
           "share_of_roof_before": round(ROOF_MS / DECODE_MS_BF16, 3),
           "bytes_after_MB": round(BYTES_AFTER_MB, 1),
           "floor_ms_after": round(BYTES_AFTER_MB / 1000.0 / 44.0 * 1000.0, 2),
           "share_of_new_floor": round((BYTES_AFTER_MB / 1000.0 / 44.0 * 1000.0) / proj_ms, 3),
           "note": "bytes-proportional model: t = sum_k f_k * t_dense / s_k over executed "
                   "weight bytes (ffn 59.04%, conv 11.38%, gqa 6.03%, lm_head+embed 13.67%), "
                   "measured per-op speedups at T=1/16 threads; vision tower and projector are "
                   "not executed at text decode and are excluded from the fractions."}
    md += ["", "## End-to-end decode projection (CPU, 16 threads, T=1)",
           "",
           f"- FFN 59.04% of bytes at {s_ffn:.2f}$\\times$ -> {f['ffn']/s_ffn:.4f} time units",
           f"- LFM3Conv 11.38% at {s_conv:.2f}$\\times$ -> {f['conv']/s_conv:.4f}",
           f"- GQA 6.03% unquantised -> {f['gqa']:.4f}; lm_head/embed 13.67% unquantised -> {f['emb']:.4f}",
           f"- total {t_units:.4f} units -> **{1/t_units:.2f}$\\times$ end-to-end** "
           f"({DECODE_MS_BF16:.1f} ms -> {proj_ms:.1f} ms per token; the post-quantisation "
           f"DRAM floor is {BYTES_AFTER_MB/44.0:.1f} ms for {BYTES_AFTER_MB:.0f} MB, so the layer "
           f"kernels run at {(BYTES_AFTER_MB/44.0)/proj_ms*100:.0f}% of the floor)"]
    with open(os.path.join(RES, "phase3_e2e.json"), "w") as fh:
        json.dump(e2e, fh, indent=1)
    with open(os.path.join(RES, "phase3_tables2.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(RES, "tab_phase3_cpu2.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    gpu = json.load(open(os.path.join(RES, "phase3_triton.json")))
    gl = ["% layer & sparsity & active & torch mv & triton dense & triton sparse & speedup & bytes read \\\\"]
    for c in gpu["cases"]:
        for lv in c["levels"]:
            gl.append(f"{c['matrix']} & {lv['tau']:.2f} & {lv['active']:.3f} & "
                      f"{lv['torch_dense_bf16']*1e3:.1f} & {lv['triton_bf16_all']*1e3:.1f} & "
                      f"{lv['triton_bf16_sparse']*1e3:.1f} & {lv['sp_bf16_vs_dense']:.2f}$\\times$ & "
                      f"{lv['bytes_read_bf16']/1e6:.2f} MB \\\\")
    with open(os.path.join(RES, "tab_phase3_gpu.tex"), "w") as fh:
        fh.write("\n".join(gl) + "\n")
    print("\n".join(md[-8:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
