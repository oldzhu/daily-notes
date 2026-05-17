Good question. Let me investigate the actual kernel path for `modelopt_fp4` in sglang to answer accurately rather than guess. 

Searched for regex `nvfp4|NVFP4|FP4|cutlass|marlin|apply.*linear|forward` (`**/python/sglang/srt/layers/quantization/modelopt_quant.py`), 20 results

Searched for regex `class.*FP4.*Method|class.*NVFP4|def apply|def forward` (`**/python/sglang/srt/layers/quantization/modelopt_quant.py`), 14 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1190-1190), lines 1190 to 1245

Searched for regex `fp4_gemm =|fp4_gemm=|cutlass_fp4_gemm|flashinfer_fp4_gemm|FP4.*support|sm.*1[02]0|sm120|capability` (`**/python/sglang/srt/layers/quantization/modelopt_quant.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#60-60), lines 60 to 155

Now I have the full picture. Let me give a clear, accurate answer:

## Short answer

**No — we are not using Marlin W4A16. The current Phase A config is pure NVFP4 → NVFP4 GEMM, no Marlin in the path.** And to your second concern: weights stay in FP4 the whole way, the activations are cast down to FP4 at runtime, and the multiply-accumulate runs on **SM120 FP4 tensor cores**. There is no BF16 dequant fallback for the weights.

## How the current setup actually dispatches

Our two SOAR profiles use two completely different LinearMethod paths:

| Profile | `--quantization` | LinearMethod | Compute kernel | Tensor cores used |
|---|---|---|---|---|
| `gptq` (production baseline) | `gptq_marlin` | `GPTQMarlinLinearMethod` | sgl-kernel **Marlin W4A16** (GPTQ INT4 weights, BF16 activations, BF16 accumulate) | BF16 (148 TF) |
| `nvfp4` (Phase A) | `modelopt_fp4` | `ModelOptFp4LinearMethod` | flashinfer `mm_fp4` / sgl-kernel `cutlass_scaled_fp4_mm` | **FP4 (593 TF)** |

So within a single run only one of those is active — never both at the same time.

## What `ModelOptFp4LinearMethod.apply` actually does

From modelopt_quant.py, the per-step path on every linear is:

1. `x_fp4, x_scale = fp4_quantize(x, layer.input_scale_inv)` — cast the BF16 activation **down to NVFP4** (FP4 E2M1, 16-element blocks, FP8 E4M3 scale).
2. `out = fp4_gemm(x_fp4, w_fp4, x_scale, w_scale, alpha, out_dtype, w_n)` — call **flashinfer `mm_fp4` (cutlass backend)** which dispatches to the SM120 FP4 cutlass GEMM. Asserts confirm: `x_fp4.dtype == uint8` (packed FP4), `weight.dtype == uint8` (packed FP4), `weight_scale.dtype == float8_e4m3fn`.
3. The kernel writes the output back in BF16/FP16 (`output_dtype`) for the next layer's input.

The weights are **never** dequantized to BF16. There's no Marlin call. The `import` chain at the top of the file (modelopt_quant.py) explicitly picks the SM120-aware code path:

```python
if is_sm120_supported():
    from flashinfer import fp4_quantize       # SM120 FP4 quantize kernel
else:
    from sgl_kernel import scaled_fp4_quant as fp4_quantize
```

and likewise modelopt_quant.py routes to `flashinfer_fp4_gemm` (cutlass backend) for the GEMM. So FP4 tensor cores **are** being used in our run — that part is fine.

## So why was Phase A so slow on long context?

Then the long-context slowness is **not** "Marlin W4A16 dequant fallback" (we don't run Marlin at all under `modelopt_fp4`). The likely actual causes, in priority order:

1. **Per-token activation re-quantization overhead.** Every linear casts activations BF16→FP4 at every step (line `x_fp4, x_scale = fp4_quantize(x, ...)`). For short prompts this is cheap (we measured ~57 tok/s). For long context with `chunked_prefill_size=65536` and `prefill_max_requests=4`, this is a 65k×4 BF16→FP4 quantize on every linear, every chunk — could swamp the FP4 GEMM gain.
2. **NVFP4 attention path is not fused.** Q/K/V projections do FP4 GEMM, but our attention kernel (flashinfer) consumes BF16. So we still pay Q/K/V FP4→BF16 round-trip into attention, attention runs in BF16, then output projection re-quantizes BF16→FP4. Compare: champion's GLA attention stays in FP8/FP4 native paths.
3. **Quality collapse causes runaway thinking.** Uniform NVFP4 produces gibberish → mcq/qa items hit `max_tokens=65536` instead of stopping at the answer → each "slow" item is actually generating **~65k tokens** before timing out. This is probably the dominant factor for the >5 min/item observations: it's not slow tokens, it's a stuck generation loop. The "Paris→unedoc.com/abc/u.com..." degeneration we saw on the 6-token test is the same failure mode.

So cause #3 is likely most of the timeout, and only after fixing quality (Phase B FourOverSix) will we get a clean signal on whether causes #1/#2 are real bottlenecks.

## Recommendation for direction "2" (FP4 kernel investigation)

Since the FP4 tensor cores **are** being used (not bypassed via Marlin), the kernel-level investigation is narrower than I implied yesterday. The useful next probes are:

- (a) Profile a single short request with `nsys` to confirm `cutlass_scaled_fp4_mm` actually dispatches an FP4 cutlass kernel on SM120 (not a fallback emulation).
- (b) Measure `fp4_quantize` overhead vs `mm_fp4` time at typical chunk sizes — find out if quantize-on-the-fly is the bottleneck.
- (c) Run a **short-context-only** accuracy probe (just the mcq subset, capped `max_tokens=512`) to separate "model is broken" from "kernel is slow".

Want me to update CHANGE_0150_phase_a_nvfp4_baseline.en.md with this corrected analysis (the doc currently says "likely BF16 dequant fallback", which is wrong), or queue these as investigation tasks for the next fcloud session?
