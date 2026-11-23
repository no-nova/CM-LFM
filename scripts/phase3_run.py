"""Phase 3 sweep driver: runs the AVX-512 LFM3Conv kernel across (threads, T, tau)
and records the kernel-level table. Single model never loaded here (weights are the
layer-0 slab dumped by scripts/phase3_dump.py), so the RAM rule is trivially met.
"""
import json, os, subprocess, sys

ROOT = "/home/user/Downloads/CM-LFM3"
BIN = f"{ROOT}/artifacts/p3conv"
DATA = f"{ROOT}/artifacts/p3"
ENV = dict(os.environ, OMP_WAIT_POLICY="passive", OMP_PLACES="cores", OMP_PROC_BIND="close")

rows = []
for threads in (1, 16):
    for T, off in ((1, 32), (16, 16), (64, 0)):
        for tau_cal in (0.0, 0.30, 0.50):
            env = dict(ENV, OMP_NUM_THREADS=str(threads))
            out = subprocess.run([BIN, DATA, str(threads), "7", "6", str(T), str(off), str(tau_cal)],
                                 env=env, capture_output=True, text=True).stdout
            blob = json.loads(out[out.index("{"):out.rindex("}") + 1])
            blob["threads"] = threads
            blob["tau_cal"] = tau_cal
            rows.append(blob)
            print(f"th={threads} T={T} tau={tau_cal}: {blob['ms']} speedup={blob['speedup']} "
                  f"active={blob['skipped']['frac_blocks_active']:.3f}", flush=True)

json.dump(rows, open(f"{ROOT}/results/phase3_kernels.json", "w"), indent=1)

# markdown table for the report
lines = ["| thr | T | tau | unfused bf16 | fused bf16 | iq4 dense | iq4 sparse | active | fused/unfused | iq4s/bf16 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    m, s = r["ms"], r["speedup"]
    lines.append(f"| {r['threads']} | {r['T']} | {r['tau_cal']:.2f} | {m['unfused_bf16']:.3f} | {m['fused_bf16']:.3f} | "
                 f"{m['fused_iq4']:.3f} | {m['fused_iq4_sparse']:.3f} | {r['skipped']['frac_blocks_active']:.3f} | "
                 f"{s['fused_vs_unfused']:.3f} | {s['iq4_sparse_vs_fused_bf16']:.3f} |")
open(f"{ROOT}/results/phase3_kernels.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
