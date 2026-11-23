"""Phase 2 machinery: runtime co-activation permutation + 32-lane block gating.

Design decisions (see results/PHASE2_REPORT.md):
  * The permutation lives in *index tensors*, never in the weights, so a
    block-gated layer with every block kept is bit-identical to the dense layer.
  * Block energy  E_b = ||a_b||_2  over 32 lanes vs a per-layer, per-budget
    threshold theta_l(tau) calibrated offline (quantile of the calibration
    block-energy distribution), plus per-token norm compensation
    alpha = min(1.5, ||a|| / ||a_keep||).
  * The gated activation is the *intermediate* one: h = silu(w1 x) * w3 x for
    the FFN (d_ff = 7680), and y = C * dwconv(B * X) for LFM3Conv (d = 2048,
    pre-out_proj).  Both are captured inside the patched forward, so the
    calibration sees exactly the tensor the kernel gates.
  * Quality is evaluated masked-dense: zeroing an inactive block is
    mathematically identical to skipping it.  Real skipping is Phase 3.
"""
import torch
import torch.nn.functional as F

DB = 32  # lanes per block


class CFG:
    on = False
    mode = "ffn"        # ffn | conv | both
    method = None       # native_topk | perm_topk | cats | teal | wanda | ours
    tau = 0.0
    variant = "A"       # A (state-exact) | B (gate-first)
    alpha = False       # R3: alpha = 1 (cancelled by the sandwich RMSNorm anyway)
    pt = False          # per-token adaptive threshold instead of a layer-global one
    rms = False         # R3: per-token RMS block energy (T-independent threshold)
    wnorm = False       # R3: weight lanes by the output column norm


ST = {}   # id(module) -> {"pi","inv","theta":{tau:float}}   (active gated modules)
REC = {}  # id(module) -> {"cov": tensor|None, "energy": list|None, "cap": int}


def block_energy(x, nb):
    return x.reshape(*x.shape[:-1], nb, DB).pow(2).sum(-1)


def block_energy_rms(x, nb, w=None):
    """Per-token RMS block energy: E[t,b] = sqrt(mean_{i<32} (a_i * w_i)^2).

    RMS (not sum) so that the calibrated threshold does not depend on the
    sequence length; w is the static output column norm ||W_out[:, i]||_2, which
    is folded in offline and therefore costs nothing at runtime.
    """
    if w is not None:
        x = x * w
    return x.reshape(*x.shape[:-1], nb, DB).pow(2).mean(-1).sqrt()


def _mask_from_energy(e, nb, tau, theta=None):
    if theta is not None:
        return e > theta
    k = min(nb, max(1, int(round((1.0 - tau) * nb))))
    idx = e.topk(k, dim=-1).indices
    return torch.zeros_like(e, dtype=torch.bool).scatter_(-1, idx, True)


def gate_mask(e, nb, st):
    """Block mask for energies e (.., nb) under the current CFG policy."""
    if CFG.method in ("ours", "teal"):
        if CFG.pt:
            # per-token adaptive threshold (analysis variant; the deployment path
            # uses the calibrated layer-global theta, never a runtime quantile)
            return e.float() > torch.quantile(e.float(), CFG.tau, dim=-1, keepdim=True)
        th = st["theta"]
        if CFG.tau not in th:
            raise KeyError(f"no theta for tau={CFG.tau}; have {sorted(th)}")
        return e > th[CFG.tau]
    return _mask_from_energy(e, nb, CFG.tau)


def _expand(m, inv):
    """(.., nb) permuted-order mask -> (.., d) native-order mask."""
    return m.repeat_interleave(DB, dim=-1).index_select(-1, inv)


def _alpha(e, mf):
    return (e.sum(-1).sqrt() / (e * mf).sum(-1).sqrt().clamp_min(1e-12)).clamp_max(1.5)


# --------------------------------------------------------- calibration record
def _collect(rec, act, st):
    """Store covariance (native order) and/or sampled block energies."""
    if rec.get("cov") is not None:
        a = act.reshape(-1, act.shape[-1]).float()
        rec["cov"].add_(a.T @ a)
    if rec.get("rows") is not None:
        got = sum(t.shape[0] for t in rec["rows"])
        if got < rec.get("cap", 4096):
            flat = act.reshape(-1, act.shape[-1])
            rec["rows"].append(flat[:rec["cap"] - got].detach().float().cpu())
    if rec.get("energy") is not None and st is not None:
        nb = act.shape[-1] // DB
        e = block_energy(act.index_select(-1, st["pi"]), nb).reshape(-1, nb)
        cap = rec.setdefault("cap", 4096)
        got = sum(t.shape[0] for t in rec["energy"])
        if got < cap:
            rec["energy"].append(e[:cap - got].detach().float().cpu())


def rec_rows(rec):
    return torch.cat(rec["energy"], 0) if rec.get("energy") else None


def reset_rec(modules, cov=False, energy=False, cap=4096, dim=None, device="cuda", rows=False):
    for m in modules:
        REC[id(m)] = {"cov": torch.zeros(dim, dim, dtype=torch.float32, device=device) if cov else None,
                      "energy": [] if energy else None, "rows": [] if rows else None, "cap": cap}


# ------------------------------------------------------------------- forwards
def ffn_forward(self, x, **_):
    h = F.silu(self.w1(x)) * self.w3(x)
    st, rec = ST.get(id(self)), REC.get(id(self))
    if rec is not None:
        _collect(rec, h, st)
    if st is None or not CFG.on:
        return self.w2(h)
    nb = h.shape[-1] // DB
    hp = h.index_select(-1, st["pi"])
    e = (block_energy_rms(hp, nb, st.get("wperm") if CFG.wnorm else None)
         if CFG.rms else block_energy(hp, nb))
    m = gate_mask(e, nb, st)
    mf = m.to(h.dtype)
    if CFG.method == "ours" and CFG.alpha:
        mf = mf * _alpha(e, mf).unsqueeze(-1)
    return self.w2(h * _expand(mf, st["inv"]))


def conv_forward(self, hidden_states=None, cache_params=None, seq_idx=None, **_):
    """Verbatim Lfm3ShortConv maths in plain PyTorch (path-matched baseline).

    Gating is per token and per channel; tensors are (b, d, T) internally, so
    masks built on (b, T, d) are transposed before the elementwise product.
    Full-sequence only: the incremental (cache) path is the Phase-3 target.
    """
    if seq_idx is not None:
        raise RuntimeError("conv_forward: packed-document (seq_idx) path not patched")
    x = hidden_states
    seq = x.shape[1]
    bcx = self.in_proj(x).transpose(-1, -2)                 # (b, 3d, T)
    d = bcx.shape[1] // 3
    B, C, X = bcx[:, :d], bcx[:, d:2 * d], bcx[:, 2 * d:]
    st, rec = ST.get(id(self)), REC.get(id(self))
    nb = d // DB
    keep = None
    if st is not None and CFG.on and CFG.variant == "B":
        eC = block_energy(C.transpose(-1, -2), nb)          # (b, T, nb)
        mC = gate_mask(eC, nb, st)
        keep = _expand(mC.to(C.dtype), st["inv"]).transpose(1, 2)   # (b, d, T)
        keepb = mC                                                  # (b, T, nb)
        B, C, X = B * keep, C * keep, X * keep
    h = B * X
    if cache_params is not None:
        # exact rolling short-conv state, as in modeling_lfm3.py:404-410
        h = cache_params.update_conv_state(h, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
        y = F.conv1d(h, self.conv.weight, self.conv.bias, padding=self.conv.padding[0], groups=d)[..., -seq:]
    else:
        y = F.conv1d(h, self.conv.weight, self.conv.bias, padding=self.conv.padding[0], groups=d)[..., :seq]
    y = C * y
    yt = y.transpose(-1, -2)                                # (b, T, d)
    if rec is not None:
        _collect(rec, yt, st)
    if st is not None and CFG.on:
        yp = yt.index_select(-1, st["pi"])
        e = (block_energy_rms(yp, nb, st.get("wperm") if CFG.wnorm else None)
             if CFG.rms else block_energy(yp, nb))
        m = gate_mask(e, nb, st)
        mf = m.to(y.dtype)
        if keep is not None:
            mf = mf * keepb
        if CFG.method == "ours" and CFG.alpha:
            mf = mf * _alpha(e, mf).unsqueeze(-1)
        y = y * _expand(mf, st["inv"]).transpose(1, 2)
    return self.out_proj(y.transpose(-1, -2).contiguous())


def patch(ffn_cls, conv_cls):
    ffn_cls.forward = ffn_forward
    conv_cls.forward = conv_forward


# ------------------------------------------------------------- permutation fit
def _perm(members):
    pi = members.reshape(-1)
    inv = torch.empty_like(pi)
    inv[pi] = torch.arange(pi.numel(), device=pi.device)
    return pi, inv


def build_perm_gram(C, W, nb):
    """Greedy coupled Gram-error blocks.

    Dropping block S costs E||sum_{i in S} a_i W_i||^2 = sum_{i,j in S} M_ij with
    M = C (*) (W W^T).  Blocks grow from the most energetic unassigned channel by
    repeatedly adding the channel with the smallest marginal cost/energy gain
    (2 sum_{i in S} M_ij + M_jj) / C_jj.
    """
    d = C.shape[0]
    M = C * (W.t() @ W)          # W is (d_out, d): M_ij = <W[:, i], W[:, j]>
    diagM, diagC = M.diagonal().clone(), C.diagonal().clone()
    alive = torch.ones(d, dtype=torch.bool, device=C.device)
    members = torch.empty(nb, DB, dtype=torch.long, device=C.device)
    acc = torch.zeros(d, device=C.device)
    for b in range(nb):
        s = int(torch.argmax(torch.where(alive, diagC, torch.full_like(diagC, -1.0))))
        members[b, 0] = s
        alive[s] = False
        acc += M[s]
        for t in range(1, DB):
            score = torch.where(alive, (2.0 * acc + diagM) / diagC.clamp_min(1e-12),
                                torch.full_like(diagC, float("inf")))
            j = int(torch.argmin(score))
            members[b, t] = j
            alive[j] = False
            acc += M[j]
    return _perm(members)


def build_perm_coact(C, nb):
    """CATS-style co-activation clustering (activation-only)."""
    d = C.shape[0]
    diagC = C.diagonal().clone()
    alive = torch.ones(d, dtype=torch.bool, device=C.device)
    members = torch.empty(nb, DB, dtype=torch.long, device=C.device)
    for b in range(nb):
        s = int(torch.argmax(torch.where(alive, diagC, torch.full_like(diagC, -1.0))))
        members[b, 0] = s
        alive[s] = False
        row = C[s].clone()
        for t in range(1, DB):
            j = int(torch.argmax(torch.where(alive, row, torch.full_like(row, -1e30))))
            members[b, t] = j
            alive[j] = False
            row = row + C[j]
    return _perm(members)


def set_perm(mod, pi, inv=None):
    d = pi.numel()
    if inv is None:
        inv = torch.empty_like(pi)
        inv[pi] = torch.arange(d, device=pi.device)
    ST[id(mod)] = {"pi": pi, "inv": inv, "theta": {}}


def identity(d, device="cuda"):
    return torch.arange(d, device=device, dtype=torch.long)


def theta_for(tau, energies):
    vals = torch.cat([e.reshape(-1) for e in energies]).float()
    return float(torch.quantile(vals, tau, interpolation="linear"))
