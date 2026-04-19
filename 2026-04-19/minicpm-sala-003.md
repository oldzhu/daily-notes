Excellent catch — you've identified a **critical distinction**. The champion post is talking about **NVFP4 KV Cache** (quantizing KV cache values to FP4), NOT **NVFP4 weight quantization** (what we did).

Let me verify this by reading the full post and checking sglang's FP4 KV cache support:  



Searched for regex `fp4.*kv|kv.*fp4|nvfp4|fp4_e2m1` (`**/python/sglang/**/*.py`), 20 results

Fetched https://mp.weixin.qq.com/s/w1g3njB24rxLCiCLxWFD7Q

This confirms it completely. You're 100% right — **we did the wrong thing**. Here's the breakdown:

## What the champion did vs what we did

| Aspect | Champion (智算一队) | Us (team-beta) |
|--------|-------------------|----------------|
| **What's FP4** | **KV Cache** values (`--kv-cache-dtype fp4_e2m1`) | **Model weights** (W4A4 quantization) |
| **Weights** | W4A16 GPTQ+Marlin (kept as-is) | FP4 E2M1 (replaced GPTQ) |
| **Accuracy impact** | 80% → 75% (only 5% drop) | 79% → 12% (catastrophic) |
| **Strategy** | Mixed-precision KV: sensitive layers keep FP8, others use FP4 | Uniform FP4 on all weights |

The champion's approach:
1. **Kept GPTQ W4A16 weights unchanged**
2. Only compressed the **KV cache** from FP8 → FP4 (halves KV memory bandwidth in decode)
3. Even that dropped accuracy 80% → 75%, so they did **per-layer sensitivity analysis** — keeping first/last layers at FP8, middle layers at FP4
4. This mixed-precision KV cache gave near-FP4 speed with near-FP8 accuracy

**SGLang already supports this!** I found:
- `--kv-cache-dtype fp4_e2m1` flag exists (model_runner.py)
- `KVFP4QuantizeUtil` class for FP4 KV quantize/dequantize
- Test files for KV FP4

This is a completely different (and much more promising) optimization direction. We should:
1. Try `--kv-cache-dtype fp4_e2m1` with our existing GPTQ model (trivial — just change one server arg)
2. If accuracy drops too much, implement per-layer mixed-precision KV cache like the champion did

Shall I prepare a proposal for testing FP4 KV cache on the GPTQ baseline? It's literally a one-arg change in `prepare_env.sh`.
