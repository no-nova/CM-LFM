#!/usr/bin/env python3
"""Phase 1 text-quality baselines, three complementary protocols.

P1  raw WikiText-2 test PPL, sliding window 1024 / stride 512 over all 294,512
    tokens (standard, restarts context every 512 tokens).
P2  continuous PPL: one pass with a KV cache, no window restarts, so every token
    after the first 256 is scored with its true causal context (the LM's attention
    span is 256, so this is the protocol that matches the architecture).
P3  chat-conditioned PPL: the paragraph is reproduced by the assistant inside the
    real chat template, scored on the assistant span only (in-distribution for a
    chat-tuned model; the raw protocols are off-distribution).

Also emits a per-window NLL profile so we can quantify the restart penalty.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_baseline import RES, get_wikitext2, ppl as ppl_sliding, MemProbe  # noqa

M = "/home/user/Downloads/CNN-CM/models/LiquidAI/LFM3-VL-2B"


@torch.inference_mode()
def continuous_ppl(model, ids, chunk=512, skip=256):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config.text_config)
    tot, cnt = 0.0, 0
    for beg in range(0, ids.shape[1], chunk):
        seg = ids[:, beg:beg + chunk]
        if seg.shape[1] < 2:
            break
        out = model(input_ids=seg, past_key_values=cache, use_cache=True)
        lg = out.logits.float()
        del out
        n0 = skip if beg == 0 else 0
        lp = torch.log_softmax(lg[0, n0:-1], -1)
        tgt = seg[0, n0 + 1:]
        tot += -lp.gather(-1, tgt[:, None]).sum().item()
        cnt += tgt.numel()
        del lg, lp
    torch.cuda.empty_cache()
    return {"ppl": float(np.exp(tot / cnt)), "tokens": cnt, "nll_mean": tot / cnt,
            "bits_per_token": tot / cnt / np.log(2)}


@torch.inference_mode()
def chat_ppl(model, proc, text_tokens, n_win=32, W=256):
    tok = proc.tokenizer
    tot, cnt = 0.0, 0
    for i in range(n_win):
        seg = text_tokens[i * W:(i + 1) * W]
        prefix = tok.decode(seg[:32])
        suffix = tok.decode(seg[32:])
        msgs = [{"role": "user", "content": [{"type": "text", "text":
                 "Continue the following text exactly as it appears, without commentary.\n\n" + prefix}]},
                {"role": "assistant", "content": [{"type": "text", "text": suffix}]}]
        full = proc.apply_chat_template(msgs, tokenize=False)
        # assistant span starts after the last "assistant" header
        head, _, tail = full.rpartition("assistant\n")
        lead = head + "assistant\n"
        ids_full = tok(lead + tail, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        ids_pre = tok(lead, return_tensors="pt", add_special_tokens=False).input_ids
        n0 = ids_pre.shape[1]
        lg = model(input_ids=ids_full, use_cache=False).logits.float()
        lp = torch.log_softmax(lg[0, n0 - 1:-1], -1)
        tgt = ids_full[0, n0:]
        tot += -lp.gather(-1, tgt[:, None]).sum().item()
        cnt += tgt.numel()
    return {"ppl": float(np.exp(tot / cnt)), "tokens": cnt, "bits_per_token": tot / cnt / np.log(2)}


@torch.inference_mode()
def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    probe = MemProbe()
    proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0},
        trust_remote_code=True).eval()
    text = get_wikitext2(os.path.join(RES, "wikitext2_test.txt"))
    ids = proc.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    tids = ids[0].tolist()

    out = {"tokens_total": int(ids.shape[1])}
    out["P1_sliding_1024_512"] = ppl_sliding(model, proc.tokenizer, text)
    torch.cuda.empty_cache()
    out["P2_continuous_16k"] = continuous_ppl(model, ids[:, :16384])
    torch.cuda.empty_cache()
    out["P3_chat_conditioned"] = chat_ppl(model, proc, tids)
    # per-window NLL profile: cost of restarting context (32 windows of the test set)
    prof = []
    for beg in range(0, 32 * 512, 512):
        lg = model(input_ids=ids[:, beg:beg + 512], use_cache=False).logits.float()
        tgt = ids[0, beg + 1:beg + 512]
        l = -torch.log_softmax(lg[0, :-1], -1).gather(-1, tgt[:, None]).squeeze(-1)
        prof.append({"pos0_31": float(l[:32].mean()), "pos32_255": float(l[32:256].mean()),
                     "pos256_511": float(l[256:].mean())})
    out["restart_penalty_nll_by_position"] = {
        k: round(float(np.mean([p[k] for p in prof])), 4) for k in prof[0]}
    out["mem_peak_mb"] = probe.close()
    json.dump(out, open(os.path.join(RES, "phase1_ppl.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
