Short answer: **theoretically yes (4× the BF16 TFLOPS), but in practice it's a high-risk path that we already tried once and it failed catastrophically — so I'd rank it BELOW the recommended #1/#3/#4/#6, not above.**

## Why it could help (the upside)

| Path | Tensor-core peak on SM120 |
|---|---|
| BF16 HMMA (today) | 148 TFLOPS |
| FP8 QMMA (W4A8 / mxfp8) | 296 TFLOPS |
| **FP4 QMMA (nvfp4 / mxfp4)** | **593 TFLOPS** |

If we could run weights and activations through native FP4 QMMA, GEMM throughput would be ~4× the current path. SM120 hardware is *built* to favor this — Blackwell explicitly added FP4 MMA, and on long-context, prefill-bound workloads (which is what the official speed set is) GEMM still matters.

## Why it's risky (the downside, with our actual data)

1. **We already ran this experiment**: Test 21 was an NVFP4 quantization attempt. Result: accuracy collapsed to ~12 %. That's far below the 97 % cliff (C=0). We have empirical evidence that naive offline NVFP4 on this model breaks it.
2. **Why NVFP4 broke MiniCPM-SALA where GPTQ INT4 worked**:
   - INT4 + per-group BF16 scale (group_size=128) → fine-grained per-group rescaling, well-suited to mean-zero Laplacian-like LLM weights.
   - NVFP4 (E2M1) → 1 sign + 2 exp + 1 mantissa, 16 distinct values, with a single shared block-scale per ~16-element block. Logarithmic spacing is great for activations with large dynamic range, but worse for **weights** which cluster near zero.
   - SALA in particular has 24 lightning layers whose state evolves multiplicatively — small per-layer FP4 errors compound across 1000+ decode steps in a recurrence.
3. **Recovery requires QAT or a heavily engineered calibration pipeline**:
   - Pure post-training NVFP4 (what Test 21 did) is documented in literature as needing **rotation transforms (QuaRot/SpinQuant) + outlier handling**, otherwise sub-7B models lose >5 pts on MMLU-class tasks.
   - QAT requires training data + GPU time we don't have for the SOAR window.
4. **Lightning kernels don't natively consume FP4**: even if weights are FP4, the SimpleGLA Triton kernels would need an FP4-aware rewrite. Marlin upstream has W4A4 prototypes, but they're FP4-act/INT4-weight, not what we'd want.
5. **Submission size constraint cuts both ways**: NVFP4 weights are smaller than INT4 (4 bits with shared block-scale ≈ 4.5 bits/elt vs INT4 with group_size=128 BF16 scale ≈ 4.125 bits/elt), so almost no win on the 2 GB cap either.

## Where it sits in the priority list

Updated priority ranking:

| Rank | Vector | Expected gain | Risk | Effort |
|---|---|---|---|---|
| 1 | W4A8 (FP8 activations, FP8 QMMA path) | ~1.5-2× on prefill GEMM | Med | High |
| 2 | Lightning state FP8 (proposal #3) | 10-20% on long-context decode | Med | Med |
| 3 | Fused lightning kernel (proposal #4) | 5-15% on decode | Low | Med |
| 4 | Speculative decoding (proposal #6) | 1.3-2× on S₁ | Low | Med-High |
| **5** | **NVFP4 weights** | **~4× GEMM peak, but…** | **Very high (Test 21 failed)** | **High (needs QuaRot/SpinQuant + retest)** |

## Recommendation

Treat NVFP4 as a **stretch goal after we have W4A8 working**:

- Step 1 (proposal #1): Get W4A8 (FP8 activations + INT4 weights via Marlin's W4A8 path). This already gets us onto QMMA hardware at 296 TF. If it works, we've captured 2/3 of the achievable speedup with much lower accuracy risk.
- Step 2 (only after W4A8 succeeds): Reconsider NVFP4 with proper outlier handling (QuaRot rotation pre-quant, then NVFP4 calibration). Budget separately because it's a multi-day calibration cycle.

**Don't** swap GPTQ INT4 → NVFP4 as the *primary* next iteration — Test 21 already paid for that lesson.

Want me to formalize this as part of the next proposal queue (e.g., add it as "Vector #7 (deferred)" in the research note), or expand any of these points?Continued with Autopilot: Task not yet complete


