"""Bug 1 audit: is the block mask dropping COLD blocks?

Re-derives, from the frozen calibration tensors, the fraction of block energy that
survives the mask at every tau, for every ordering, under BOTH decision rules
(calibrated threshold and fixed top-K).  Three invariants are asserted:

 I1  retained(tau) > 1 - tau                      (can't lose more energy than the
                                                   fraction of weights you keep)
 I2  retained_coact(tau) >= retained_native(tau)  (the co-activation clustering must
                                                   concentrate energy, or the ordering
                                                   is pointing the wrong way)
 I3  retained_gram(tau) ~= retained_native(tau)   (the Gram coupling is energy-neutral
                                                   by construction -- if it is not,
                                                   the covariance and the weights are
                                                   out of phase)

Any failure exits non-zero, so the Phase-2 pipeline can gate on it.
"""
from __future__ import annotations

import json
import os
import sys

import torch

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load():
    return torch.load(os.path.join(RES, "phase2_calibration.pt"), map_location="cpu", weights_only=False)


def retained(E: torch.Tensor, theta: float) -> float:
    """E: (rows, nb) squared per-token block energies.  Threshold rule, as at runtime."""
    keep = E > theta
    denom = E.sum().clamp_min(1e-30)
    return float((E * keep).sum() / denom)


def retained_topk(E: torch.Tensor, tau: float) -> float:
    nb = E.shape[1]
    k = min(nb, max(1, int(round((1.0 - tau) * nb))))
    top = torch.topk(E, k, dim=-1).values
    return float(top.sum() / E.sum().clamp_min(1e-30))


def main() -> int:
    cal = load()
    taus = [0.10, 0.20, 0.30, 0.40, 0.50]
    fails: list[str] = []
    rows = []
    for fam in ("ffn", "conv"):
        mods = [k for k in cal if k.endswith("." + fam)]
        for tau in taus:
            rec = {"family": fam, "tau": tau}
            for order in ("native", "gram", "coact"):
                E = cal[mods[0]][f"E_{order}"]
                th = cal[mods[0]][f"theta_{order}"][tau]
                r_th = retained(E, th)
                r_tk = retained_topk(E, tau)
                rec[f"{order}_thr"] = r_th
                rec[f"{order}_topk"] = r_tk
                if not r_th > 1.0 - tau - 1e-9:
                    fails.append(f"I1 {fam} {order} tau={tau}: retained_thr={r_th:.4f} <= {1-tau:.2f}")
                if not r_tk > 1.0 - tau - 1e-9:
                    fails.append(f"I1 {fam} {order} tau={tau}: retained_topk={r_tk:.4f} <= {1-tau:.2f}")
            if rec["coact_thr"] + 1e-6 < rec["native_thr"]:
                fails.append(f"I2 {fam} tau={tau}: coact {rec['coact_thr']:.4f} < native {rec['native_thr']:.4f}")
            if abs(rec["gram_thr"] - rec["native_thr"]) > 0.02:
                fails.append(f"I3 {fam} tau={tau}: gram {rec['gram_thr']:.4f} vs native {rec['native_thr']:.4f}")
            rows.append(rec)

    # per-layer robustness: worst-case retained ratio at tau=0.5 over all 37 modules
    worst = {"ffn": 1.0, "conv": 1.0}
    for k, v in cal.items():
        fam = k.split(".")[-1]
        if fam not in worst:
            continue
        worst[fam] = min(worst[fam], retained(v["E_coact"], v["theta_coact"][0.50]))

    print(f"{'fam':5s} {'tau':5s} " + " ".join(f"{o:>9s}" for o in ("nat_thr", "gram_thr", "coact_thr", "nat_topk", "coact_topk", "1-tau")))
    for r in rows:
        print(f"{r['family']:5s} {r['tau']:<5.2f} "
              f"{r['native_thr']:9.4f} {r['gram_thr']:9.4f} {r['coact_thr']:9.4f} "
              f"{r['native_topk']:9.4f} {r['coact_topk']:9.4f} {1-r['tau']:9.4f}")
    print("\nworst-case per-layer retained energy at tau=0.50 (coact):", worst)

    out = {"rows": rows, "worst_coact_tau50": worst, "failures": fails, "pass": not fails}
    with open(os.path.join(RES, "phase2_energy_audit.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n{'PASS' if not fails else 'FAIL'}  ({len(fails)} violations)")
    for m in fails[:20]:
        print("   ", m)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
