Good observations — the speed did improve with dense calibration. Let me investigate your questions: (1) whether `sparse_qkv_w8` preset still correctly assigns W8 to the right QKV layers when `sparse_config=None`, (2) what other calibration/accuracy levers we have. 

Created 4 todos

Research sparse_qkv_w8 preset mechanism

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#200-200), lines 200 to 400

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#400-400), lines 400 to 600

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#600-600), lines 600 to 800

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#800-800), lines 800 to 1050

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1050-1050), lines 1050 to 1104

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#100-100), lines 100 to 164

Searched for text `mixer_types` (`**/benchmark/soar/**`), 20 results

Searched for regex `damp_percent|damp_auto_increment|desc_act|sym|static_groups` (`**/benchmark/soar/**`), 20 results

Searched for regex `damp_percent|damp_auto_increment|QuantizeConfig` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 3 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/configs/minicpm.py#30-30), lines 30 to 167

Searched for regex `mixer_types.*minicpm4|mixer_types.*lightning` (`**/benchmark/soar/demo_sala/**`), 5 results

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/gptq_presets_comprehensive.md)

Searched for regex `damp_percent|damp_auto|desc_act|true_sequential|static_groups|mse|perplexity|divergence|accuracy.*analysis` (`**/benchmark/soar/demo_sala/**`), 20 results

Searched for regex `class QuantizeConfig`, no results

Ran terminal command: from gptqmodel import QuantizeConfig; import inspect; print(inspect.signature(QuantizeConfig.__init__))" 2>/dev/null || echo "gptqmodel not installed locally

Searched modelcloud/gptqmodel for "QuantizeConfig class __init__ damp_percent desc_act sym static_groups true_sequential parameters", 64 results

Starting: *Analyze accuracy divergence distribution* (2/4)

Excellent findings. Let me also check the actual Test 24 accuracy breakdown more carefully to understand where the divergence is: 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#80-80), lines 80 to 180

Now let me look at the detailed accuracy results to understand the divergence pattern:  

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 55

Let me check how calibration data is prepared and the GPTQModel quantize call:   

Searched for regex `damp_percent|damp_auto|desc_act|mse|act_group_aware|true_sequential` (`**/benchmark/soar/demo_sala/preprocess_model.py`), no results

Searched for regex `model\.quantize|QuantizeConfig|quant_config` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 10 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#880-880), lines 880 to 960

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#760-760), lines 760 to 900

Now I have a clear picture. Let me also check what GPTQModel's QuantizeConfig accepts:    

Searched for regex `_build_dynamic_rules` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 3 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#400-400), lines 400 to 530

Starting: *Identify accuracy improvement levers* (3/4)

Now let me check the accuracy test breakdown across tests to understand variance:   

Searched for regex `mcq.*50\.00|mcq.*56\.67|mcq.*63\.33` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), 2 results

Starting: *Propose testing plan* (4/4)

Here's the complete analysis:

---

## Answers to Your Questions

### 1. Yes, `sparse_qkv_w8` still correctly assigns W8 under dense calibration

The W8 assignment is based on `config.mixer_types` (which layers are "standard attention" vs "lightning"), **NOT** on `sparse_config`. Setting `sparse_config=None` (CHANGE_0110) only affects the attention *pattern* (sparse vs dense) during calibration — it does NOT change which layers get W8 quantization. The 8 full-attention layers (indices 0,9,16,17,22,29,30,31) still get their QKV at W8, and the 24 lightning layers stay at W4.

### 2. Accuracy Variance Analysis

Looking at historical mcq results across tests with the **exact same code and config**:
- Test 18: mcq=**50.00%** → Test 18b (re-run): mcq=**63.33%** (same commit a9f4d43cb!)
- Test 20: mcq=63.33%, Test 23: mcq=56.67%, Test 24: mcq=50.00%

MCQ has only 30 samples — each sample is worth 3.33%. The swing between 50% and 63.33% is just 4 questions. **MCQ is the most volatile metric** and drives most of the accuracy/C fluctuation.

### 3. Calibration Tuning Levers Available

| Lever | Current Value | Options to Try | Potential Impact |
|-------|--------------|----------------|-----------------|
| `damp_percent` (GPTQModel) | 0.05 (default) | 0.01, 0.1, 0.15 | Lower → more aggressive quantization (faster but less accurate); Higher → more regularized (slower but more stable) |
| `mse` optimization | 0 (disabled) | 0.01-0.1 | Enables MSE grid search for optimal scale/zero — can improve per-layer accuracy at cost of longer calibration |
| More calibration samples | 90 | 150+ (add NIAH samples?) | More data → better Hessian estimation → better quantization grid |
| Calibration sequence length | Variable (up to 128K) | Truncate to 32K? | Shorter sequences reduce sparse/dense mismatch effect |

### 4. Proposed Testing Plan

Given the high variance (especially mcq ±13%), we need **multiple runs of the same config** to distinguish signal from noise:

**Phase A: Establish variance baseline (3 runs of sparse-calibrated baseline)**
- Re-quantize with `SOAR_GPTQ_FORCE_DENSE=0` (sparse calibration, the proven baseline)
- Run accuracy test 3 times to establish the true variance range
- This tells us whether Test 24's drop is dense-calibration effect or just variance

**Phase B: Test dense calibration with accuracy-improving levers**
- Test B1: Dense calibration + `damp_percent=0.1` (more regularization)
- Test B2: Dense calibration + `mse=0.05` (MSE grid search)  
- Test B3: Dense calibration + more calibration samples (add NIAH to the 90)

**Phase C: Compare best configs**
- Pick best from Phase B, run 3 times for variance analysis
- Compare against Phase A baseline

Shall I proceed? Specifically:
1. Should I first **revert to sparse calibration, re-quantize, and run 3 accuracy tests** (Phase A) to establish the variance baseline?
2. Or do you want to jump directly to trying `damp_percent` and `mse` tuning on dense calibration (Phase B)?
3. How many fcloud sessions are you comfortable running? Each re-quantization takes ~85 minutes, each accuracy test ~50 minutes. 

Completed: *Propose testing plan* (4/4)


