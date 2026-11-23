# LFM3-VL-2B architecture anatomy (measured from the checkpoint)

* text: d=2048, d_ff=7680, k=7, 24 layers, 16 heads x 128 = 2048 (q/o), 4 KV heads (512), vocab 128000, tied embeddings, sliding window 256
* layer_types: convx13, full_attentionx6, sliding_attentionx5
* vision: d=1024, d_ff=2560, 14 layers, patch 16, temporal 2, merge 2; types: convx8, full_attentionx6
* total params (excl. vision) 1,656,037,888; total checkpoint bytes 3658.6 MiB

| component | params (M) | bf16 MiB | share |
|---|---|---|---|
| text FFN (24x) | 0.00 | 2160.0 | 59.04% |
| text LFM3Conv block (13x) | 0.00 | 416.4 | 11.38% |
| vision LFM3Conv+attn+FFN | 0.00 | 313.2 | 8.56% |
| text GQA attention (11x) | 0.00 | 220.7 | 6.03% |
| multimodal projector | 0.00 | 48.0 | 1.31% |
| embed_tokens (tied) | 262.14 | 500.0 | 13.67% |

## Arithmetic intensity α_l = FLOPs / bytes moved (bf16, weights streamed once)

| operator | α (T=1) | α (T=64) | weight MiB |
|---|---|---|---|
| q_proj/o_proj (d x d) | 1.0 | 60.24 | 8.00 |
| k_proj/v_proj (512 x d) | 1.0 | 51.2 | 2.00 |
| ffn.w1/w3 (d_ff x d) | 1.0 | 62.95 | 30.00 |
| ffn.w2 (d x d_ff) | 1.0 | 62.95 | 30.00 |
| conv.in_proj (3d x d) | 1.0 | 62.69 | 24.00 |
| conv.out_proj (d x d) | 1.0 | 60.24 | 8.00 |
