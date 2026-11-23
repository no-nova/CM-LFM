"""Phase 4 stage-1 calibration: derives TAU_ABS and the global-motion noise floor
from a clip known to be static (V1_static_surveillance), so the skipping thresholds
are calibrated rather than hand-tuned per clip.
"""
import json, sys
import numpy as np

sys.path.insert(0, "/home/user/Downloads/CM-LFM3/scripts")
from phase4_video import load_clip, frame_diffs, GRID

frames = load_clip("V1_static_surveillance")
d = frame_diffs(frames)[1:]
res = {
    "clip": "V1_static_surveillance (8 frames, 224x224, 32x32 blocks)",
    "tau_abs_p95": round(float(np.quantile(d, 0.95)), 6),
    "tau_abs_p99": round(float(np.quantile(d, 0.99)), 6),
    "noise_mean": round(float(d.mean()), 6),
    "noise_max": round(float(d.max()), 6),
}
json.dump(res, open("/home/user/Downloads/CM-LFM3/results/phase4_calib.json", "w"), indent=1)
print(json.dumps(res, indent=1))
