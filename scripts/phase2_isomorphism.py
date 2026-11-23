#!/usr/bin/env python3
"""Phase 2, step 1: numerical verification of the two isomorphic permutations.

  pi_ffn  : intermediate-channel permutation of a SwiGLU FFN
            w1 <- w1[pi], w3 <- w3[pi], w2 <- w2[:, pi]        (exact by construction)
  pi_conv : channel permutation of the LFM3Conv block
            in_proj rows [0:d], [d:2d], [2d:3d] each permuted,
            conv.weight[pi], out_proj[:, pi]

Checks:
  A. single operator in float64 (CPU, random weights): must be exactly 0
  B. single operator in float32 (CPU): must be <= 1e-6
  C. whole model, bf16, 128-token prefill (GPU): logits must be bit-identical
  D. negative control: permute in_proj but not out_proj -> error must be large
     (proves the test is sensitive, i.e. C is not vacuous)
"""
import json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B")
from lfm3_conv import lfm3_causal_conv1d                      # noqa: E402
from transformers import AutoModelForImageTextToText          # noqa: E402

RES = "/home/user/Downloads/CM-LFM3/results"
M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"
D, DFF, K = 2048, 7680, 7


# ---------------------------------------------------------------- operator level
def ffn_ref(x, w1, w3, w2):
    return w2(F.silu(w1(x)) * w3(x))


def ffn_perm(w1, w3, w2, pi):
    return w1[pi], w3[pi], w2[:, pi]


class ConvRef(torch.nn.Module):
    """Stand-alone copy of modeling_lfm3.Lfm3ShortConv maths (no Triton)."""

    def __init__(self, d, k, dtype):
        super().__init__()
        self.in_proj = torch.nn.Linear(d, 3 * d, bias=False, dtype=dtype)
        self.out_proj = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        self.conv = torch.nn.Conv1d(d, d, k, groups=d, bias=True, dtype=dtype)

    def forward(self, x, perm=None):
        bcx = self.in_proj(x).transpose(-1, -2)              # (d, 3T)
        b, c, h = bcx.chunk(3, dim=-2)
        h = b * h
        y = F.conv1d(h, self.conv.weight, self.conv.bias, padding=K - 1,
                     groups=h.shape[0])[..., :x.shape[0]]
        y = (c * y).transpose(-1, -2).contiguous()
        return self.out_proj(y)


def permute_conv(m, pi):
    d = m.in_proj.out_features // 3
    with torch.no_grad():
        m.in_proj.weight = torch.nn.Parameter(
            torch.cat([m.in_proj.weight[pi], m.in_proj.weight[d + pi],
                       m.in_proj.weight[2 * d + pi]]), requires_grad=False)
        m.conv.weight = torch.nn.Parameter(m.conv.weight[pi], requires_grad=False)
        m.conv.bias = torch.nn.Parameter(m.conv.bias[pi], requires_grad=False)
        m.out_proj.weight = torch.nn.Parameter(m.out_proj.weight[:, pi], requires_grad=False)


def operator_checks():
    out = {}
    torch.manual_seed(0)
    pi = torch.randperm(DFF)
    x = torch.randn(32, D)
    for dt, nm in ((torch.float64, "float64"), (torch.float32, "float32")):
        w1 = torch.randn(DFF, D, dtype=dt) * 0.02
        w3 = torch.randn(DFF, D, dtype=dt) * 0.02
        w2 = torch.randn(D, DFF, dtype=dt) * 0.02
        y0 = ffn_ref(x.to(dt), lambda z: z @ w1.T, lambda z: z @ w3.T, lambda z: z @ w2.T)
        a, b, c = ffn_perm(w1, w3, w2, pi)
        y1 = ffn_ref(x.to(dt), lambda z: z @ a.T, lambda z: z @ b.T, lambda z: z @ c.T)
        out[f"ffn_{nm}"] = {"max_abs": float((y0 - y1).abs().max()),
                            "rel_l2": float((y0 - y1).norm() / y0.norm())}
    for dt, nm in ((torch.float64, "float64"), (torch.float32, "float32")):
        torch.manual_seed(1)
        m = ConvRef(D, K, dt).eval()
        pi = torch.randperm(D)
        x = torch.randn(16, D, dtype=dt)
        with torch.no_grad():
            y0 = m(x)
            m2 = ConvRef(D, K, dt).eval()
            m2.load_state_dict(m.state_dict())
            permute_conv(m2, pi)
            y1 = m2(x)
            # negative control: gate/branch permutation without the out_proj column permutation
            m3 = ConvRef(D, K, dt).eval()
            m3.load_state_dict(m.state_dict())
            with torch.no_grad():
                d = D
                m3.in_proj.weight = torch.nn.Parameter(
                    torch.cat([m3.in_proj.weight[pi], m3.in_proj.weight[d + pi],
                               m3.in_proj.weight[2 * d + pi]]))
                m3.conv.weight = torch.nn.Parameter(m3.conv.weight[pi])
                m3.conv.bias = torch.nn.Parameter(m3.conv.bias[pi])
            y2 = m3(x)
        out[f"conv_{nm}"] = {"max_abs": float((y0 - y1).abs().max()),
                             "rel_l2": float((y0 - y1).norm() / y0.norm()),
                             "negative_control_max_abs": float((y0 - y2).abs().max())}
    return out


# ---------------------------------------------------------------- whole model
@torch.inference_mode()
def model_checks(n_tok=128):
    model = AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    lm = model.model.language_model
    ids = torch.randint(0, 120000, (1, n_tok), device="cuda")
    ref = model(input_ids=ids).logits.float()
    layers = lm.layers
    conv_layers = [l for l in layers if hasattr(l, "conv")]
    ffn_layers = [l for l in layers if hasattr(l, "mlp") or hasattr(l, "feed_forward")]
    res = {"n_conv_layers": len(conv_layers), "n_ffn_layers": len(ffn_layers),
           "seq_len": n_tok}
    g = torch.Generator(device="cpu").manual_seed(7)

    def ffn_of(l):
        return getattr(l, "mlp", None) or getattr(l, "feed_forward")

    # -- pi_ffn on every FFN
    saved = []
    with torch.no_grad():
        for l in ffn_layers:
            f = ffn_of(l)
            pi = torch.randperm(f.w1.out_features, generator=g).cuda()
            saved.append((f, pi))
            f.w1.weight = torch.nn.Parameter(f.w1.weight[pi])
            f.w3.weight = torch.nn.Parameter(f.w3.weight[pi])
            f.w2.weight = torch.nn.Parameter(f.w2.weight[:, pi])
    out = model(input_ids=ids).logits.float()
    res["pi_ffn_all"] = {"max_abs": float((out - ref).abs().max()),
                         "rmse": float((out - ref).pow(2).mean().sqrt()),
                         "top1_agree": float((out.argmax(-1) == ref.argmax(-1)).float().mean())}
    with torch.no_grad():   # undo
        for f, pi in reversed(saved):
            inv = torch.argsort(pi)
            f.w1.weight = torch.nn.Parameter(f.w1.weight[inv])
            f.w3.weight = torch.nn.Parameter(f.w3.weight[inv])
            f.w2.weight = torch.nn.Parameter(f.w2.weight[:, inv])
    assert (model(input_ids=ids).logits.float() - ref).abs().max() == 0, "undo failed"

    # -- pi_conv on every conv layer
    cc = []
    with torch.no_grad():
        for l in conv_layers:
            c = l.conv
            pi = torch.randperm(c.in_proj.out_features // 3, generator=g).cuda()
            cc.append((c, pi))
            d3 = c.in_proj.out_features
            c.in_proj.weight = torch.nn.Parameter(
                torch.cat([c.in_proj.weight[pi], c.in_proj.weight[D + pi],
                           c.in_proj.weight[2 * D + pi]]))
            c.conv.weight = torch.nn.Parameter(c.conv.weight[pi])
            c.out_proj.weight = torch.nn.Parameter(c.out_proj.weight[:, pi])
    out = model(input_ids=ids).logits.float()
    res["pi_conv_all"] = {"max_abs": float((out - ref).abs().max()),
                          "rmse": float((out - ref).pow(2).mean().sqrt()),
                          "top1_agree": float((out.argmax(-1) == ref.argmax(-1)).float().mean())}
    with torch.no_grad():
        for c, pi in reversed(cc):
            inv = torch.argsort(pi)
            w = c.in_proj.weight
            c.in_proj.weight = torch.nn.Parameter(torch.cat([w[inv], w[D + inv], w[2 * D + inv]]))
            c.conv.weight = torch.nn.Parameter(c.conv.weight[inv])
            c.out_proj.weight = torch.nn.Parameter(c.out_proj.weight[:, inv])
    out = model(input_ids=ids).logits.float()
    res["undo_after_conv_max_abs"] = float((out - ref).abs().max())

    # -- both at once
    with torch.no_grad():
        for l in ffn_layers:
            f = ffn_of(l)
            pi = torch.randperm(f.w1.out_features, generator=g).cuda()
            f.w1.weight = torch.nn.Parameter(f.w1.weight[pi])
            f.w3.weight = torch.nn.Parameter(f.w3.weight[pi])
            f.w2.weight = torch.nn.Parameter(f.w2.weight[:, pi])
        for l in conv_layers:
            c = l.conv
            pi = torch.randperm(c.in_proj.out_features // 3, generator=g).cuda()
            c.in_proj.weight = torch.nn.Parameter(
                torch.cat([c.in_proj.weight[pi], c.in_proj.weight[D + pi],
                           c.in_proj.weight[2 * D + pi]]))
            c.conv.weight = torch.nn.Parameter(c.conv.weight[pi])
            c.out_proj.weight = torch.nn.Parameter(c.out_proj.weight[:, pi])
    out = model(input_ids=ids).logits.float()
    res["pi_ffn_and_conv"] = {"max_abs": float((out - ref).abs().max()),
                              "rmse": float((out - ref).pow(2).mean().sqrt()),
                              "top1_agree": float((out.argmax(-1) == ref.argmax(-1)).float().mean())}
    res["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return res


if __name__ == "__main__":
    out = {"operator_level": operator_checks()}
    out["whole_model_bf16"] = model_checks()
    json.dump(out, open(os.path.join(RES, "phase2_isomorphism.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
