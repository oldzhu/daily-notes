the 90 qa-cwe-mcq samples does not improve the accuracy:

--- Evaluation Results ---

Average Score: 78.04%
Total Duration: 2823.02 s
Total Tokens: In=8644166, Out=1235142
Average Tokens/Sample: In=57627.8, Out=8234.3
Overall TPS (Output): 437.52 tokens/s

Per-task Accuracy
  cwe: count=30 correct=24.4 accuracy=81.33% avg_in=74163.4 avg_out=20094.3
  fwe: count=30 correct=29.6667 accuracy=98.89% avg_in=68153.7 avg_out=9377.7
  mcq: count=30 correct=15.0 accuracy=50.00% avg_in=269.9 avg_out=9081.9
  niah: count=30 correct=29.0 accuracy=96.67% avg_in=73982.7 avg_out=2517.6
  qa: count=30 correct=19.0 accuracy=63.33% avg_in=71569.1 avg_out=100.0

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=15.0 accuracy=50.00% avg_in=269.9 avg_out=9081.9
  len_32k_128k: count=80 correct=67.6 accuracy=84.50% avg_in=92784.4 avg_out=8555.5
  len_4k_32k: count=40 correct=34.4667 accuracy=86.17% avg_in=30332.8 avg_out=6956.1

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=15.6 accuracy=78.00% avg_in=95471.0 avg_out=20080.1
  task=cwe|len_4k_32k: count=10 correct=8.8 accuracy=88.00% avg_in=31548.2 avg_out=20122.6
  task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=10458.1
  task=fwe|len_4k_32k: count=10 correct=9.6667 accuracy=96.67% avg_in=29697.8 avg_out=7216.8
  task=mcq|len_0_4k: count=30 correct=15.0 accuracy=50.00% avg_in=269.9 avg_out=9081.9
  task=niah|len_32k_128k: count=20 correct=19.0 accuracy=95.00% avg_in=95307.5 avg_out=3586.5
  task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=379.9
  task=qa|len_32k_128k: count=20 correct=13.0 accuracy=65.00% avg_in=92977.5 avg_out=97.4
  task=qa|len_4k_32k: count=10 correct=6.0 accuracy=60.00% avg_in=28752.2 avg_out=105.1

  I am  think mixed-precision quantization, for example:

# 创建混合精度配置
mixed_config = {
    "attention.q_proj": {"bits": 4, "group_size": 128},
    "attention.k_proj": {"bits": 4, "group_size": 128},
    "attention.v_proj": {"bits": 4, "group_size": 128},
    "attention.o_proj": {"bits": 8, "group_size": 128},  # 输出投影使用更高精度
    "mlp.gate_proj": {"bits": 3, "group_size": 64},
    "mlp.up_proj": {"bits": 3, "group_size": 64},
    "mlp.down_proj": {"bits": 4, "group_size": 128},
    "lm_head": {"bits": 8, "group_size": -1},  # LM头保持较高精度
    "norm": {"bits": 16},  # 归一化层保持FP16
}

could research and imvestigate what kind of mixed-precssion could balance the accuracy and bechmark efficiency and porivide the action plan we can move forward.
=========
I’m checking the latest SOAR rule text and the current GPTQ mixed-precision hooks in this repo before proposing a next step, so the plan is both compliant and actually implementable.



Fetched 2 resources

Search complete: Inspect mixed precision support

Searched for files matching `**/preprocess_model.py`, 1 match

Searched for files matching `**/gptqmodel_minicpm_sala.py`, 1 match

Searched for files matching `**/benchmark/soar/demo_sala/**`, 200 matches

Searched for regex `GPTQ|quantization|mixed precision|group_size|per-module|bits`, 200 results

Searched for files matching `**/gptq.py`, 2 matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/base_config.py#1-1), lines 1 to 200

Searched for text `packed_modules_mapping` (`**/minicpm.py`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#130-130), lines 130 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#750-750), lines 750 to 823

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 150

I have the rule and capability baseline. I’m reading the exact mixed-precision override points in the current code so the action plan can distinguish “already supported with small glue code” from “would require loader/runtime work.”

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#402-402), lines 402 to 460

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#744-744), lines 744 to 820

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils.py#245-245), lines 245 to 320

I’m checking whether your example’s `3-bit` MLP idea is even compatible with the current `gptq_marlin` path here, because that is likely the biggest practical constraint on the mixed-precision menu.

Searched for regex `TYPE_MAP|weight_bits not in \[2, 3, 4, 8\]|Unsupported quantization config` (`**/python/sglang/srt/layers/quantization/gptq.py`), 14 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#120-120), lines 120 to 175

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#330-330), lines 330 to 380

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#218-218), lines 218 to 288

I checked the latest SOAR competition and toolkit pages on April 2. The current rules still support quantization-path optimization, require on-platform preprocessing rather than shipping a quantized model, and keep the correctness gate strict: below 97% relative accuracy, the total score is 0. The toolkit 技术路径指引 still points directly at GPTQ W4A16 + Marlin + FP8 KV cache as the primary quantization route, so a mixed-precision GPTQ experiment is compliant, but it has to stay compatible with the current loader/runtime path.

Your 90-sample result is useful because it changes the diagnosis. It strongly suggests the problem is no longer calibration coverage. You fully covered qa, mcq, and cwe, but overall accuracy stayed around 78. That means the remaining loss is much more likely to be structural quantization sensitivity than sampling variance. The mcq drop is especially important: it is concentrated in the short bucket, so this is not just a long-context K/Q routing problem.

My recommendation is to pursue fused-group-safe mixed precision, not arbitrary per-module mixed precision.

What is feasible in the current codebase:
- The underlying GPTQ stack already supports per-module dynamic overrides for bits and group_size in gptq.py and utils.py.
- The current SOAR preprocess wrapper does not expose those positive overrides yet. In preprocess_model.py, _build_dynamic_rules only emits negative skip rules, not per-module mixed-precision rules.
- The current benchmark defaults in preprocess_model.py and prepare_env.sh only support include/exclude scope, not mixed bit-width presets.

What is not a good first mixed-precision target:
- Your example’s 3-bit MLP idea is not a good fit for the current gptq_marlin path. In gptq.py, GPTQMarlinConfig only supports 4-bit and 8-bit symmetric formats. So 3-bit would likely force a different runtime path and hurt speed or compatibility.
- Norm layers do not need special handling in this pipeline. In gptqmodel_minicpm_sala.py, norms are already marked as native weights.
- lm_head is also not the first place to spend complexity. It is not in the current selected GPTQ scope, so the more useful baseline is to keep it native rather than explicitly pushing it to 8-bit.
- Per-shard experimentation inside Q, K, and V is still risky because MiniCPM serves through merged qkv_proj at runtime. Even when quantized, that fused boundary must stay format-consistent.

Best first mixed-precision candidates, ranked:
1. Raise all self_attn.o_proj modules from W4 to W8, keep qkv and MLP at W4.
2. If that is too expensive, raise only selected layers’ self_attn.o_proj to W8, for example the last 8 layers or only the sparse/full-attention layers.
3. If o_proj-only helps but not enough, test self_attn.o_proj plus mlp.down_proj at W8 for a narrow layer subset.
4. Keep qkv fused groups and gate/up fused groups uniform at W4 in the first iteration.

Why I recommend o_proj first:
- It is a separate module, not part of the fused qkv loader boundary.
- It sits after attention routing, so it can affect both long-context qa/cwe behavior and short-context mcq behavior.
- It is already called out in the MiniCPM GPTQ model support as a shape-sensitive module in gptqmodel_minicpm_sala.py, which makes it a plausible accuracy lever.
- It avoids the exact loader incompatibility we just hit around qkv.

Action plan I recommend for the next approved feature:
- Objective:
  Add positive dynamic GPTQ overrides so we can run fused-group-safe mixed precision while preserving the current gptq_marlin serving path.
- Exact files to change:
  preprocess_model.py
  prepare_env.sh
  New bilingual docs under docs/soar_2026_changes as the next change pair
- Exact implementation shape:
  Add a new env-driven dynamic override input, for example a JSON string or compact preset format, and merge it into the existing exclude-based dynamic rules.
  Keep the existing include/exclude behavior unchanged.
  Add one first preset only: W8 for self_attn.o_proj, W4 for everything else already quantized.
  Do not change qkv, gate_proj, or up_proj bit-widths in the first experiment.
- Validation commands:
  bash -n prepare_env.sh
  python3 -m py_compile preprocess_model.py
  bash prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
  python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_DIR> --data_path <DATA_DIR>/perf_public_set.jsonl --concurrency 32
  bash SOAR/bench_serving.sh http://127.0.0.1:30000
- Success criteria:
  Accuracy recovers meaningfully above the current 78-level baseline, especially in mcq and qa.
  Serving remains loadable with gptq_marlin.
  Speed regression is modest enough that the correctness coefficient gain outweighs the duration loss.
- Failure criteria:
  Accuracy remains flat within noise.
  Speed regresses sharply at S1/S8 without clear correctness recovery.
  Any load/runtime incompatibility appears around fused modules.

My assessment of likely payoff:
- Highest-probability safe test: W8 self_attn.o_proj only.
- Highest-risk item from your sketch: 3-bit gate_proj/up_proj under gptq_marlin.
- Lowest-value complexity for now: explicit lm_head and norm mixed-precision tuning.

If you want, I can turn this into the next formal proposal as one feature:
CHANGE_0065: GPTQ mixed precision via W8 attention output projection
and keep the scope limited to the two benchmark files plus the required EN/ZH docs, with no code changes until you approve.
