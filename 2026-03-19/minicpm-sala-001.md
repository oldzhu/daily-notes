I’m narrowing this to the args that actually matter for your MiniCPM-SALA submission path and the specific questions you asked, rather than trying to enumerate hundreds of flags mechanically. I’ll pull the authoritative definitions for backend selection, KV cache dtype, FP8 GEMM backend, and the main scheduler/cudagraph knobs, then summarize what is worth testing.

Searched for regex `attention-backend|prefill-attention-backend|decode-attention-backend|fp8-gemm-backend|kv-cache-dtype|dense-as-sparse|force-dense-minicpm|schedule-conservativeness|prefill-max-requests|max-running-requests|chunked-prefill-size|skip-server-warmup|disable-cuda-graph|cuda-graph-bs|enable-piecewise-cuda-graph` (`**/python/sglang/srt/**`), 116 results

Searched for files matching `**/python/sglang/srt/server_args.py`, 2 matches

Searched for files matching `**/benchmark/soar/demo_sala/prepare_env.sh`, 1 match

I’ve got the relevant definitions. I’m reading the exact help text and a couple of validation branches so I can answer the relation questions precisely, especially for attention backends, KV cache dtype, FP8 GEMM backend, and CUDA graph.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_utils.py#180-180), lines 180 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/attention_registry.py#210-210), lines 210 to 245

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#2660-2660), lines 2660 to 2875

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#3360-3360), lines 3360 to 3445

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#3990-3990), lines 3990 to 4148

I have the core definitions. I’m checking one more runtime override path so I don’t tell you a backend combination is valid when the model runner will silently override it for MiniCPM-SALA.

Searched for regex `minicpm_flashinfer|minicpm_flashattn|prefill_attention_backend|decode_attention_backend|attention_backend` (`**/python/sglang/srt/model_executor/**`), 80 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#1600-1600), lines 1600 to 1665

Trying to explain every flag in that help dump one by one would not be useful. There are hundreds, and most are irrelevant to your current SOAR path.

The right way to look at `launch_server` for your MiniCPM-SALA submission is:

1. a small set of high-impact args actually controls speed
2. some args are simple fallbacks for others
3. some args only matter for completely different model families or inference modes

For your current path, the important args are the ones already visible in prepare_env.sh, plus a few closely related backend options.

**What matters for your current submission**
Your active path is basically:
- `gptq_marlin`
- `fp8_e5m2` KV cache
- CUDA graph enabled
- `minicpm_flashinfer`
- `force-dense-minicpm`
- `dense-as-sparse`
- chunked prefill
- conservative scheduler tuning

So the flags worth understanding are:

1. Quantization / numeric path
- `--quantization`
- `--kv-cache-dtype`
- `--dtype`
- `--quantization-param-path`

2. Attention backend selection
- `--attention-backend`
- `--prefill-attention-backend`
- `--decode-attention-backend`

3. Scheduler / memory / batch shape
- `--mem-fraction-static`
- `--max-running-requests`
- `--chunked-prefill-size`
- `--prefill-max-requests`
- `--max-prefill-tokens`
- `--schedule-conservativeness`

4. CUDA graph
- `--skip-server-warmup`
- `--cuda-graph-bs`
- `--disable-cuda-graph`
- `--disable-cuda-graph-padding`
- `--enable-piecewise-cuda-graph`

5. MiniCPM-specific switches
- `--dense-as-sparse`
- `--force-dense-minicpm`
- `--fuse-topk`
- `--split-stage1`

6. FP8 GEMM backend
- `--fp8-gemm-backend`

Everything else in that huge help output is mostly for other architectures, multimodal pipelines, distributed topologies, speculative decoding, MoE, LoRA, disaggregation, etc.

**Relations Between Attention Backend Args**
This part is clear in server_args.py and model_runner.py.

The relationship is:

- `--attention-backend`
  - global default for attention backend

- `--prefill-attention-backend`
  - overrides the global backend for prefill only

- `--decode-attention-backend`
  - overrides the global backend for decode only

So effectively:
- if you only set `--attention-backend`, both prefill and decode use it
- if you set `--prefill-attention-backend`, prefill uses that instead
- if you set `--decode-attention-backend`, decode uses that instead
- if prefill and decode differ, SGLang creates a hybrid attention backend at runtime

So to your exact question: you cannot really have “3 independent backends” in the sense of three active choices at once. The global one is a fallback. The actual active runtime backend count is either:
- one backend for both phases
- or two backends: one for prefill, one for decode

And yes, you can set prefill and decode to different things. The code supports that in model_runner.py. But it also warns that this hybrid mode is experimental.

For your MiniCPM path, I would be conservative:
- only split prefill and decode if you have a clear hypothesis and a clean A/B plan
- otherwise keep one backend for both phases

**What These Important Args Do**
Here is the practical explanation of the important ones.

**1. `--quantization`**
Controls the weight quantization method.
For you:
- `gptq_marlin` is the important one
- this is the main reason your dense path is fast

Changing this is not a light tuning knob. It changes the core inference kernel path.

**2. `--kv-cache-dtype`**
Controls how KV cache is stored.
From server_args.py, the main useful options for you are:
- `auto`
- `bf16`
- `fp8_e5m2`
- `fp8_e4m3`

Practical meaning:
- `bf16`: safest for accuracy, slowest / heaviest bandwidth
- `fp8_e5m2`: more dynamic range, usually safer numerically for long-context serving
- `fp8_e4m3`: more mantissa precision, less range, sometimes better raw precision in narrow ranges but less robust when activations vary a lot

For MiniCPM-SALA long-context inference, `fp8_e5m2` is usually the safer default. I would not assume `fp8_e4m3` is “better” overall. It is only “better” if your KV values stay in a range where the extra mantissa matters more than the lost exponent range.

So for your question:
- no, `fp8_e4m3` is not generally better than `fp8_e5m2` for inference
- for this long-context competition path, `fp8_e5m2` is the safer choice unless a controlled test proves otherwise

**3. `--attention-backend`**
Chooses the kernel implementation for attention layers.

This is high impact, but only if the model/backend combination actually supports the path cleanly.

Your current path uses:
- `minicpm_flashinfer`

That is already a specialized choice for MiniCPM.

**4. `--prefill-attention-backend` / `--decode-attention-backend`**
These let you split backend choice by phase:
- prefill: large prompt processing
- decode: token-by-token generation

This can matter because the fastest backend for long prefill is not always the fastest backend for decode.

But in your case, I would treat this as a medium-risk experiment, because:
- mixed backend mode is explicitly warned as experimental in model_runner.py
- MiniCPM-SALA has hybrid attention behavior, so backend interactions are more fragile than for a plain dense decoder

**5. `--fp8-gemm-backend`**
This is defined in server_args.py and dispatched in fp8_utils.py.

This only affects Blockwise FP8 GEMM operations.

That is a critical distinction:
- your current main linear path is `gptq_marlin`, which is W4A16 Marlin
- so `--fp8-gemm-backend` is not the main driver for your current quantized dense path

In other words:
- yes, changing `--fp8-gemm-backend` can impact performance
- but for your current `gptq_marlin` submission, it is probably not one of the highest-ROI knobs
- it matters much more for FP8 linear kernels, not your main GPTQ Marlin GEMMs

So I would not spend time there first.

**6. `--mem-fraction-static`**
Controls how much GPU memory is reserved for static allocations such as weights and KV pool.

Practical effect:
- too low: you leave performance on the table or reduce effective KV capacity
- too high: you risk OOM or runtime instability

This is high impact and worth tuning carefully.

**7. `--max-running-requests`**
Upper bound on active requests.

Practical effect:
- higher can improve throughput
- too high can hurt graph capture, scheduling stability, or memory pressure

This is one of the most important throughput knobs.

**8. `--chunked-prefill-size`**
Maximum prompt chunk size during chunked prefill.

Practical effect:
- larger can help long prefill throughput
- too large can hurt memory stability or capture behavior

This is important for long-context workloads like SOAR.

**9. `--prefill-max-requests`**
Maximum number of requests merged into one prefill batch.

Practical effect:
- higher can improve throughput if prefill is underutilized
- too high can hurt latency, memory pressure, or correctness stability indirectly through serving behavior

You already saw that this knob can change behavior materially.

**10. `--max-prefill-tokens`**
Upper bound on total prefill tokens in one batch.

This interacts with:
- `chunked-prefill-size`
- `prefill-max-requests`

These three should be thought of together, not independently.

**11. `--schedule-conservativeness`**
Controls how aggressively the scheduler packs / retracts work.

Practical effect:
- smaller: more aggressive, can improve throughput
- larger: safer, less retraction churn

You already saw that over-tuning this can backfire.

**12. `--skip-server-warmup`**
Skips warmup.

Practical effect:
- faster startup
- but worse initial graph/capture preparation, and can affect early behavior

For official benchmarking, this can matter if startup-state behavior leaks into measurement or readiness.

**13. `--disable-cuda-graph`**
Turns off CUDA graph.

This is usually a bad idea for your competition path.
The repo itself warns of “huge performance loss” in cuda_graph_runner.py.

For your path:
- keep CUDA graph on unless debugging

**14. `--cuda-graph-bs`**
Controls which batch sizes get graphed.

This is a potentially useful knob if you have evidence that official workload batch distribution differs from local workload.

**15. `--disable-cuda-graph-padding`**
Avoids graph use when padding would be required.

This can help if padding waste is hurting you, but it can also reduce graph coverage. Medium-risk tuning knob.

**16. `--enable-piecewise-cuda-graph`**
Experimental prefill/extend optimization.

For your current MiniCPM-SALA path, I would not prioritize this without a clear compatibility check. Earlier investigation already suggested it is not a clean drop-in.

**17. `--dense-as-sparse`**
MiniCPM-specific behavior switch.

In your current force-dense setup, you already found this does not materially matter much under `--force-dense-minicpm`.

**18. `--force-dense-minicpm`**
This is the high-value MiniCPM-specific knob in your recent work.
It forces dense attention behavior and has been the safer official baseline path.

**Can any args still help speed up?**
Yes, but not all equally.

For your exact path, I would rank remaining argument-based opportunities like this:

1. `--cuda-graph-bs`
Why:
- official and local shape mix may differ
- graph coverage mismatch is a plausible reason why local speed intuition does not transfer

2. `--max-running-requests`
Why:
- strong effect on throughput and graph capture shape coverage
- but must be tuned conservatively

3. `--chunked-prefill-size`
Why:
- official workload has many long prompts
- this directly affects long-prompt serving behavior

4. `--max-prefill-tokens`
Why:
- interacts with chunked prefill and request merging

5. `--prefill-max-requests`
Why:
- high leverage, but you already saw it can destabilize behavior

6. `--prefill-attention-backend` / `--decode-attention-backend`
Why:
- possible upside if prefill and decode prefer different kernels
- but this is more experimental and riskier than scheduler/graph tuning

I would rank these lower for your current branch:
- `--fp8-gemm-backend`
- `--kv-cache-dtype=fp8_e4m3`
- `--disable-cuda-graph`
- `--enable-piecewise-cuda-graph`

**Direct answers to your specific questions**
**Can I set prefill, decode, and attention backend to different things?**
- You can set prefill and decode differently.
- `attention-backend` is the fallback default.
- If both phase-specific backends are set, the global one is effectively irrelevant.

**Is `fp8_e4m3` better than `fp8_e5m2`?**
- Not generally.
- `fp8_e4m3` has more mantissa precision, less range.
- `fp8_e5m2` has more range, usually safer for long-context KV cache.
- For your MiniCPM-SALA competition path, `fp8_e5m2` is the safer default.

**Would changing `fp8-gemm-backend` affect performance?**
- Yes, but mainly for FP8 GEMM paths.
- For your current `gptq_marlin` main path, it is probably not one of the most important speed knobs.

**My concrete recommendation**
If you want speedup from args alone, without changing code, I would focus on only two experiment families next:

1. CUDA graph coverage tuning
- `--cuda-graph-bs`
- maybe `--disable-cuda-graph-padding`

2. Prefill scheduling envelope tuning
- `--max-running-requests`
- `--chunked-prefill-size`
- `--max-prefill-tokens`
- `--prefill-max-requests`

And I would keep these fixed for now:
- `--quantization gptq_marlin`
- `--kv-cache-dtype fp8_e5m2`
- `--force-dense-minicpm`
- `--attention-backend minicpm_flashinfer`

If you want, the next step I can do is prepare a very small arg-tuning proposal with exactly 2 or 3 controlled launch-profile variants from the current safe baseline, instead of touching code again.
