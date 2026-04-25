Yes — **closed-loop optimize→profile→optimize→profile** is exactly the right methodology. Each cycle:
1. Profile to identify the bottleneck
2. Implement one targeted optimization
3. Re-profile to verify the gain and find the next bottleneck
4. Repeat

For Marlin GEMM (85.3% of prefill), let me first research what tuning levers actually exist before proposing changes.

I also notice you have an S1 speed test running — let me check that first. 

Checked terminal output

Created 3 todos

S1 speed test completed: **120.58s** (matches our baseline ~120.5s). Now let me research the Marlin GEMM tuning options. 

Research Marlin GEMM kernel

Searched for text `marlin` (`**/python/sglang/srt/layers/quantization/**`), 20 results

Searched for text `marlin` (`**/sgl-kernel/**`), 20 results

Searched for text `marlin` (`**/python/sglang/srt/layers/linear.py`), 20 results

Searched for files matching `**/marlin*`, 11 matches

Searched for text `GPTQMarlinLinearMethod` (`**/python/sglang/srt/layers/quantization/gptq.py`), 4 results

Searched for files matching `**/sgl-kernel/csrc/gemm/marlin/*`, 8 matches

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/marlin.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#80-80), lines 80 to 220

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/kernel.h)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin.cuh)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#560-560), lines 560 to 700

Searched for text `apply_gptq_marlin_linear` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#700-700), lines 700 to 900

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#100-100), lines 100 to 500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#220-220), lines 220 to 400

Searched for text `gptqmodel` (`**/python/sglang/srt/layers/quantization/**`), 10 results

Searched for regex `fp8.*quant|fp8.*linear|Fp8LinearMethod` (`**/python/sglang/srt/layers/quantization/**`), 20 results

Searched for files matching `**/quantization/fp8*`, 6 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#460-460), lines 460 to 560

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#500-500), lines 500 to 800

Searched for text `determine_exec_config` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 2 results

Searched for regex `class Fp8|Fp8LinearMethod|Fp8Config` (`**/python/sglang/srt/layers/quantization/fp8.py`), 20 results

Searched for text `class` (`**/python/sglang/srt/layers/quantization/marlin_utils_fp8.py`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#800-800), lines 800 to 1000

Searched for text `thread_n_blocks` (`**/sgl-kernel/csrc/gemm/marlin/marlin_template.h`), 5 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1000-1000), lines 1000 to 1100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils_fp8.py#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8.py#100-100), lines 100 to 250

Ran terminal command:  wc -l /home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu /home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h

Searched for text `gptq_marlin_gemm` (`**/sgl-kernel/**`), 12 results

Searched for text `gptq_marlin_gemm` (`**/sgl-kernel/**`), 12 results

Searched for text `should_use_atomic_add_reduce` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#439-439), lines 439 to 465

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1100-1100), lines 1100 to 1210

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8.py#215-215), lines 215 to 260

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/gemm.py#505-505), lines 505 to 560

Searched for text `can_auto_enable_marlin_fp8` (`**/python/sglang/srt/layers/quantization/fp8.py`), 4 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1350-1350), lines 1350 to 1453

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#1-1), lines 1 to 80

Starting: *Formulate GEMM optimization proposal* (2/3)

Excellent research. Now let me dive deeper into the SM120 auto-config scoring to understand exactly what's happening on our Blackwell GPU: 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#100-100), lines 100 to 250

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#250-250), lines 250 to 500

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#500-500), lines 500 to 700

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#700-700), lines 700 to 850

Now let me check the `marlin_mm` dispatch to understand how the kernel is launched and what `thread_m_blocks` values our model uses: 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#850-850), lines 850 to 1100

Now let me check what SM120 config specializations exist, and look at the `COMMON_GET_IF` for SM120 — I notice it only instantiates `{N=8,K=8,T=256}`, `{N=8,K=4,T=128}`, `{N=4,K=8,T=128}` and `{N=16,K=4,T=256}`. But the SM120 thread configs include `{K=128,N=256,T=256}` which maps to `N_BLOCKS=16, K_BLOCKS=8`. Let me check if that combo is actually instantiated:  

Searched for regex `16, 8, 256` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), no results

Searched for text `N_BLOCKS, K_BLOCKS, NUM_THREADS` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 10 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#530-530), lines 530 to 540

This is a critical finding. Let me check the MiniCPM-SALA model dimensions to understand the exact GEMM shapes: 

Get MiniCPM-SALA GEMM dimensions

Searched for regex `hidden_size|intermediate_size|num_attention_heads|num_key_value_heads|head_dim|vocab_size|num_hidden_layers` (`**/minicpm_sala*`), no results

Searched for text `group_size` (`**/benchmark/soar/demo_sala/**`), 20 results

Searched for files matching `**/benchmark/soar/demo_sala/*config*`, no matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Searched for regex `hidden_size|intermediate_size|num_attention_heads|num_key_value_heads|head_dim|vocab_size|num_hidden_layers` (`**/python/sglang/srt/models/minicpm_sala.py`), no results

Searched for regex `config\.` (`**/python/sglang/srt/models/minicpm_sala.py`), no results

Searched for files matching `**/minicpm_sala.py`, no matches

Searched for files matching `**/minicpm*sala*`, no matches

Searched for files matching `**/minicpm*.py`, 20 matches

Searched for regex `hidden_size|intermediate_size|num_attention_heads|num_key_value_heads|head_dim|vocab_size|num_hidden_layers|config\.` (`**/python/sglang/srt/models/minicpm3.py`), 20 results

Searched for regex `hidden_size|intermediate_size|group_size|num_attention|head_dim` (`**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/minicpm.py#1-1), lines 1 to 100

Searched for regex `hidden_size|intermediate_size|num_attention_heads|num_key_value_heads|head_dim|vocab_size|num_hidden_layers|qk_nope|qk_rope|v_head_dim` (`**/benchmark/soar/demo_sala/preprocess_model.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/minicpm.py#100-100), lines 100 to 167

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#85-85), lines 85 to 200

Searched for regex `qk_nope_head_dim|qk_rope_head_dim|v_head_dim|q_lora_rank|kv_lora_rank|num_heads|config\.num` (`**/python/sglang/srt/models/minicpm3.py`), 30 results

Searched for regex `qk_nope|qk_rope|v_head|q_lora|kv_lora|num_heads|config\.` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 18 results

Searched for regex `config\.qk_nope|config\.qk_rope|config\.v_head|config\.q_lora|config\.kv_lora|config\.num_attention|config\.hidden_size|config\.intermediate|config\.num_hidden` (`**/python/sglang/srt/models/minicpm3.py`), 20 results

Searched for regex `scale_depth|dim_model_base|scale_width|rope_theta|max_position|rope_scaling` (`**/python/sglang/srt/models/minicpm3.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#420-420), lines 420 to 560

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#270-270), lines 270 to 330

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#350-350), lines 350 to 440

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#455-455), lines 455 to 530

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1060-1060), lines 1060 to 1110

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Searched for regex `qk_nope_head_dim|qk_rope_head_dim|v_head_dim|q_lora_rank|kv_lora_rank|dim_model_base|scale_emb|scale_depth` (`**/benchmark/soar/demo_sala/preprocess_model.py`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#90-90), lines 90 to 130

Searched for regex `qk_nope_head_dim|qk_rope_head_dim|q_lora_rank|kv_lora_rank|dim_model_base|scale_emb|scale_depth` (`**/python/sglang/srt/configs/minicpm.py`), no results

Searched for regex `qk_nope_head_dim|qk_rope_head_dim|q_lora_rank|kv_lora_rank|dim_model_base|scale_emb|scale_depth|mixer_types` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/configs/minicpm.py`), 16 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#870-870), lines 870 to 900

Searched for regex `qk_nope_head_dim.*=.*\d|qk_rope_head_dim.*=.*\d|q_lora_rank.*=.*\d|kv_lora_rank.*=.*\d|dim_model_base.*=.*\d|scale_emb.*=.*\d|scale_depth.*=.*\d` (`**/*.py`), 20 results

Searched for regex `MiniCPM.*hidden_size.*4096|MiniCPM.*qk_nope.*64|MiniCPM.*qk_rope.*32|MiniCPM.*kv_lora.*256`, no results

Searched for regex `hidden_size|qk_nope|qk_rope|kv_lora_rank|q_lora_rank|dim_model_base` (`**/docs/soar_2026_changes/**`), no results

Searched for regex `hidden_size|intermediate_size|num_attention_heads|qk_nope_head_dim|qk_rope_head_dim` (`**/docs/soar_2026_changes/**`), no results

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/minicpm_attention_analysis.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/comprehensive_minicpm_speed_analysis.md)

Searched for regex `hidden_size|intermediate_size|num_attention_heads|num_key_value_heads|head_dim|vocab_size|num_hidden_layers|config\.q_lora|config\.kv_lora|qk_nope|qk_rope` (`**/python/sglang/srt/models/minicpm.py`), 15 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#170-170), lines 170 to 230

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#575-575), lines 575 to 630

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#525-525), lines 525 to 580

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#280-280), lines 280 to 380

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#695-695), lines 695 to 750

Now let me check what the model's actual GEMM dimensions map to for the SM120 tile config: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#420-420), lines 420 to 460

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#460-460), lines 460 to 530

Now I have a comprehensive understanding. Let me formulate the proposal:  

Starting: *Present proposal for approval* (3/3)

---

## Optimization Proposal: Marlin GEMM SM120 Kernel Tile Instantiation (CHANGE_0125)

### Objective
Enable the Blackwell SM120 auto-config to use its top-scoring larger tile configs for Marlin GPTQ GEMM. Currently the kernel template instantiations are missing, causing fallback to smaller tiles.

### Expected Gain
- **5-15% GEMM speedup** → **4-13% total prefill speedup** (GEMM = 85.3% of prefill)
- Potentially **3-8% decode speedup** for M=1 path

### Key Finding: Missing Kernel Template Instantiations

The SM120 auto-config (`score_sm120_candidate()`) tries 5 thread configs in priority order:

| Priority | Config (K, N, Threads) | K_blocks | N_blocks | M1 (decode) | M234 (prefill) |
|----------|----------------------|----------|----------|-------------|----------------|
| 1 | `{128, 256, 256}` | 8 | 16 | **NOT instantiated** | **NOT instantiated** |
| 2 | `{64, 256, 256}` | 4 | 16 | **NOT instantiated** | **Works** (COMMON_GET_IF_M234) |
| 3 | `{128, 128, 256}` | 8 | 8 | **Works** (COMMON_GET_IF_M1) | **NOT instantiated** |
| 4 | `{64, 128, 128}` | 4 | 8 | Works | Works |
| 5 | `{128, 64, 128}` | 4 | 4 | Works | Works |

**Result**: Decode falls to priority #3. Prefill falls to priority #2. Neither can use the top-priority `{128, 256, 256}` config because `(N_blocks=16, K_blocks=8)` is never instantiated in `get_marlin_kernel()`.

### Model GEMM Dimensions (all compatible with N=256, K=128)

| Layer | K | N | N%256 | K%128 |
|-------|------|-------|-------|-------|
| Lightning q/k/v_proj | 4096 | 1024 | ✓ | ✓ |
| Lightning o_proj | 1024 | 4096 | ✓ | ✓ |
| Std attn q_proj | 4096 | 4096 | ✓ | ✓ |
| Std attn kv_proj | 4096 | 2048 | ✓ | ✓ |
| MLP gate_up_proj | 4096 | 28672 | ✓ | ✓ |
| MLP down_proj | 14336 | 4096 | ✓ | ✓ |

### Proposed Changes

**File: gptq_marlin.cu**

Add 4 new macro lines to `COMMON_GET_IF`:
```cpp
#define COMMON_GET_IF(W_TYPE)              \
  COMMON_GET_IF_M1(W_TYPE, 16, 8, 256)    \  // NEW: SM120 top priority
  COMMON_GET_IF_M1(W_TYPE, 16, 4, 256)    \  // NEW: SM120 2nd priority for decode
  COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)     \
  COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)     \
  COMMON_GET_IF_M1(W_TYPE, 4, 8, 128)     \
  COMMON_GET_IF_M234(W_TYPE, 16, 8, 256)  \  // NEW: SM120 top priority for prefill
  COMMON_GET_IF_M234(W_TYPE, 16, 4, 256)  \
  COMMON_GET_IF_M234(W_TYPE, 8, 8, 256)   \  // NEW: SM120 3rd priority for prefill
  COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)   \
  COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)
```

Also add enhanced logging in `log_sm120_exec_config_once` to log **all** unique configs (not just the first one).

### Risk Assessment
- **Correctness**: Zero risk — same Marlin kernel code, just different tile parameters
- **Accuracy**: Zero impact — GEMM numerics unchanged
- **Stability**: Low risk — auto-config validates shared memory fit before selecting
- **Compile time**: Moderate increase (+5-10 min for sgl-kernel build due to more template instantiations)
- **Binary size**: Moderate increase (~160 new kernel specializations)

### Rule Compliance
- No accuracy impact (pure GEMM scheduling change)
- Within 2GB submission limit (binary size increase is ~KB, not GB)
- Reproducible (deterministic tile selection via auto-config scoring)

### Verification Plan
1. Build sgl-kernel with new instantiations
2. Start server — check `log_sm120_exec_config_once` output to confirm new configs selected
3. Run speed benchmark (S1/S8/Smax) — compare against Test 25 baseline
4. Run accuracy benchmark — confirm no regression
5. **Re-profile** to verify GEMM speedup and find the next bottleneck

### Alternative Path (INVESTIGATE in parallel)
**FP8 W8A16 quantization**: Replace GPTQ 4-bit with FP8 8-bit weights → uses native Blackwell FP8 tensor cores (~2× GEMM throughput). Higher impact but higher risk (accuracy, memory). Can investigate after this change.

---

**Do you approve this change?** I'll implement the kernel instantiation additions and enhanced logging, then we'll need to build sgl-kernel on fcloud to test. 

Completed: *Present proposal for approval* (3/3)

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/marlin_gemm_sm120_research.md)


