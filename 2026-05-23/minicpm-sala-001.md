Good points — let me address both.

## Your question: does `--kv-cache-dtype fp8_e5m2` force all compute to FP8?

**No.** The KV cache dtype only controls how the **K and V tensors are stored in the paged attention pool**. It's a storage format, not a compute format.

Specifically:
- **Attention compute path**: K and V are read from the FP8 pool, dequantized to bf16 (or kept fp8 inside the FlashInfer kernel for the QK / PV matmuls — depends on backend) and the attention softmax/output runs in higher precision. KV dtype affects **only attention**.
- **GEMM path (Q/K/V projection, O projection, MLP gate/up/down)**: completely independent of `--kv-cache-dtype`. These use whatever the **weight quantization scheme** dictates:
  - GPTQ → Marlin INT4-weight × bf16-activation kernel (W4A16 effectively)
  - **NVFP4 → modelopt fp4 GEMM that uses the SM120 fp4 tensor cores** (593 TFLOPS path), with bf16 activations quantized on-the-fly to fp4 inside the kernel.

So with our config (`--quantization modelopt_fp4 --kv-cache-dtype fp8_e5m2`):
- Linear layers → fp4 tensor cores ✓ (the 593 TFLOPS path you asked about)
- Attention K/V storage → fp8 ✓ (saves 2× KV memory vs bf16, allowing larger KV pool / higher concurrency at long context)
- Attention QK / softmax / PV → bf16 inside flashinfer

These are orthogonal optimizations, both active.

The only thing that would **not** use fp4 tensor cores is if a layer fell back to bf16 because it wasn't quantized (e.g. embedding, lm_head, layer norms). For SALA those are a small fraction of FLOPs.

## On FOS vs plain NVFP4

You're right — the champion article explicitly states FOS yields **stable accuracy gains** across tasks, so plain NVFP4 should be **worse**, not better. Skipping the plain-NVFP4 isolation makes sense if we trust that result.

That refocuses the diagnosis on something else for our 75.98%:
- **Calibration data / sample count** — they may have used more or different calibration samples
- **FOS scope** — they may apply FOS only to certain projections (e.g. skip certain layers), or in a different MSE formulation
- **Numerical noise** — local 150-sample eval has ±2-3pp variance (we've seen this on GPTQ baseline: Tests 29/30/33/34a all in the 75-79% band on the same config)
- **Server config differences** — chunked-prefill / mem-fraction / max-bs tuning
- **Eval dataset variance** — they tested on private+public; we only have public

So: **re-run accuracy first** to see if 75.98% repeats or floats up to 78%+ (within local noise band). If repeatable, we have a real gap to chase. Speed bench is independent and worth doing regardless.

## Suggested plan

Keeping instance time minimal:

1. **Start instance** (await your confirm).
2. **Speed bench on existing FOS ckpt** — `fcloud_workflow.py speed --variant all`. Compares against Test 12 GPTQ baseline (S1=121.71s, S8=44.09s, Smax=35.86s). Expected: notably faster (fp4 tensor cores active) on at least Smax / high concurrency where compute-bound.
3. **Re-run accuracy** — same ckpt, same server, just re-run `accuracy`. Tells us if 75.98% is floor or noise.
4. (Optional, if time permits) Restart server with a different scheduling profile to retest accuracy — e.g. drop `mixed-chunk` and lower max-running-requests, in case scheduling-induced runaway generation is biasing mcq/qa. (We saw this pattern on GPTQ tests 30/32/33.)
5. **Pause**.

Total: ~70 min (10 min speed + 45 min accuracy + 15 min margin). 

**Confirm to start the instance and proceed?** I'll wait for your "go" before any fcloud action.
