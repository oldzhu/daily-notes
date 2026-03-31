No code changes yet. I’m treating this as two queued proposals with different priority:

1. active next proposal: accuracy-first
2. parked research proposal: TurboQuant-inspired KV work, held until accuracy is stably at least `80` and lower-risk speed work is exhausted

**Proposal 1**
CHANGE_0063: preserve K projection precision during GPTQ

Objective and expected gain:
- recover `acc_ori`, especially in `qa`, `mcq`, and `cwe`
- expected mechanism: reduce attention-routing distortion by keeping `self_attn.k_proj` in higher precision during quantization
- likely score upside is larger than another small runtime tweak, because your current best official speed is already strong and correctness is the main multiplier bottleneck

Why this is the right next feature:
- your recent calibration expansions did not produce a clean improvement
- the TurboQuant evidence and your own results point in the same direction: K-side precision matters more than V-side precision
- this is a smaller and more defensible change than broad dequantization

Rule-compliance check:
- stays inside allowed preprocessing/quantization customization
- keeps the same MiniCPM-SALA base model
- preserves the current GPTQ W4A16 + Marlin path
- does not alter concurrency, prefix-cache behavior, or submission contract

Risk to accuracy/stability:
- accuracy risk is low to medium and directionally favorable
- speed risk is real, because one projection stays unquantized
- expected speed cost should be much smaller than rolling back full attention blocks or disabling GPTQ broadly

Exact files/functions to change:
- prepare_env.sh
  - change the default GPTQ module selection so `self_attn.k_proj` is excluded from quantization
- existing selector path already supports this through environment variables, so preprocess_model.py likely does not need logic changes
- bilingual docs for the feature:
  - soar_2026_changes

What success looks like:
- `qa`, `mcq`, `cwe` improve or become more stable
- `acc_ori` moves back toward `80+`
- benchmark slowdown is limited enough that score improves overall

What failure looks like:
- accuracy does not improve meaningfully
- speed loss is too large relative to the accuracy gain

Validation commands:
```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

If CHANGE_0063 is not enough, the next accuracy ladder would be:
1. preserve `k_proj` only
2. if still weak, test preserving both `q_proj` and `k_proj`
3. only after that consider broader selective rollback

**Proposal 2**
Parked research branch: asymmetric KV or V-only compression inspired by TurboQuant

Current status:
- researched and evaluated
- not recommended as the next implementation
- held until:
  - `acc_ori >= 80` is reached and reasonably stable
  - current lower-risk speed avenues are exhausted

Objective and expected gain:
- reduce KV-side bandwidth on the standard attention subset
- possible benefit mainly at long context and decode-heavy workloads

Why it is not the right next move:
- this repo currently supports KV cache storage as standard dtypes like FP8/FP4 in server_args.py and kv_cache.py
- TurboQuant is not just a dtype switch; it needs a new cache format, storage layout, quant/dequant path, and attention-kernel support
- most of your current active MiniCPM-SALA speed path is the lightning recurrent-state path in minicpm.py and hybrid_linear_attn_backend.py, not the standard KV-cache path
- so even a successful TurboQuant-style integration would not accelerate the majority path under the current force-dense setup

Key evaluation conclusion:
- TurboQuant could be complementary
- it is not a replacement for W4A16 weight quantization
- it is high-effort and high-risk in this SGLang MiniCPM-SALA fork
- if pursued later, the safer direction is asymmetric K/V or V-only work, not aggressive K compression

Exact files/functions likely involved later:
- server_args.py
- kv_cache.py
- minicpm.py
- standard attention backend and cache storage logic
- likely new kernel work in SGLang or sgl-kernel

What success would look like later:
- measurable long-context speedup on the standard attention subset
- little or no correctness loss
- no regressions on the lightning-dominant path

What failure would look like:
- no meaningful score gain
- large integration cost
- quality loss from K-side distortion
- kernel complexity without competition-relevant return

**Recommendation**
Proceed with Proposal 1 first. Hold Proposal 2 exactly as you requested.

If you want, the next step is for me to prepare and implement CHANGE_0063 as the single active feature:
- exclude `self_attn.k_proj` from GPTQ by default
- add the required EN/ZH docs
- keep everything else unchanged until you benchmark it
