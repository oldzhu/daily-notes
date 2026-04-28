Yes, I still recommend testing Plan B first.

Why Plan B first (even with 2x weight memory and lower peak TFLOPS than FP4):
1. It is already implemented and integration-ready — we can get a real measurement today.
2. The kernel (`fp8_blockwise_scaled_mm`) already exists for SM120, so there is no kernel-engineering risk.
3. Even at FP8 peak (296 TFLOPS), it still beats the current ~100–140 TFLOPS Marlin W4 path, so measurable speedup is plausible.
4. It gives us a real data point to calibrate expectations before investing weeks in Plan C.
5. Memory headroom on this GPU (84 GB) is very large relative to a ~9B model, so 2x weight footprint is acceptable.
6. If Plan B fails the gate, we learn exactly why (accuracy drift, decode BW, kernel perf) — that directly informs Plan C design.

Regarding Plan C quantization format:

Plan C keeps **W4** weights (4-bit), not FP8.

Specifically, Plan C is a "W4A8" design:
1. **Weights**: stay as 4-bit GPTQ (same memory footprint as today, no 2x blowup)
2. **Activations**: quantized to FP8 at runtime
3. **Compute**: dequantize W4 → FP8 in registers, then feed SM120 FP8 tensor cores (UMMA)
4. **Accumulate**: FP32, output BF16

So the picture is:
| | Weight bits | Activation bits | Weight memory | Kernel work |
|---|---|---|---|---|
| Current (Marlin) | 4 | 16 (BF16) | 1x | none |
| Plan B | 8 (FP8) | 8 (FP8) | ~2x | none (kernel exists) |
| Plan C | 4 (W4) | 8 (FP8) | 1x | write new CUDA kernel |

Plan C is attractive precisely because it keeps W4’s memory/bandwidth advantage (important for decode) while still unlocking SM120 FP8 tensor cores for prefill. But it requires writing a fused dequant+GEMM kernel, which is weeks of work and high risk.

Plan C does **not** use FP4. FP4 (593 TFLOPS) would be a separate "Plan D" direction — higher peak but very high accuracy risk (we already saw NVFP4 destroy accuracy in Test 21).

Bottom line:
1. Run Plan B gate test first — cheap, fast, informative.
2. If Plan B passes gate → ship it and iterate on secondary optimizations.
3. If Plan B fails gate → pivot to Plan C (W4 + FP8 activations, custom kernel). 


