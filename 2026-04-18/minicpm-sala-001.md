Created 5 todos

Good data points. Key observations:

1. **Accuracy variance confirmed**: Same package → acc_ori 78.71 vs 80.51 across submissions. C jumped 0.96→1.0 but score still dropped 52.94→51.08 because other teams' Duration_best improved (our relative performance declined).

2. **torch.compile stays** — agreed.

3. **CHANGE_0085 packaging**: I did NOT package it — the fcloud was shut down after testing. You'll need to start fcloud again so I can sync, rebuild the tarball, and download it.

4. **Concurrent worktrees**: I'll work on both EAGLE3 Phase 2 and NVFP4 in separate worktrees.

Let me update records first, then start the concurrent work. 

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now let me add the official submission result to the test tracking file and then start concurrent work: 

Searched for text `Official Submission` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), no results

Searched for regex `Official|submission|v18` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), no results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#70-70), lines 70 to 90

Replacing 3 lines with 12 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```


Starting: *Create NVFP4 git worktree* (2/5)

Now let me create the NVFP4 worktree and start both tracks. First, check existing worktrees:  

Ran terminal command:  cd /home/oldzhu/sglang && git worktree list

Ran terminal command:  cd /home/oldzhu/sglang && git branch -a | grep -i nvfp4
Completed with input:  cd /home/oldzhu/sglang && git branch -a | grep -i nvfp4

Ran terminal command:  cd /home/oldzhu/sglang && git worktree add /home/oldzhu/sglang-nvfp4 -b nvfp4-weight-quant HEAD

Starting: *Research NVFP4 weight quant implementation* (3/5)

Now let me research NVFP4 weight quantization in depth to understand feasibility. The key questions are:
1. Does SGLang/sgl-kernel already support NVFP4 (W4A16 with FP4 E2M1 format)?
2. What CUDA kernels are needed (vs GPTQ+Marlin)?
3. How does GPTQModel support NVFP4?
4. What's needed for MiniCPM-SALA specifically?  

Research NVFP4 weight quantization support

Searched for regex `nvfp4|fp4|e2m1` (`**/python/sglang/srt/**`), 50 results

Searched for regex `modelopt|gptqmodel|GPTQ` (`**/python/sglang/srt/**`), 50 results

Searched for text `torchao` (`**/python/sglang/srt/**`), 30 results

Searched for regex `blackwell|sm_120|sm120|compute_12` (`**/sgl-kernel/**`), 30 results

Searched for files matching `**/python/sglang/srt/layers/quantization/*.py`, 50 matches

Searched for text `marlin fp4 nvfp4` (`**/python/sglang/srt/layers/quantization/**`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/modelopt_utils.py)

Searched for files matching `**/python/sglang/srt/layers/quantization/modelopt_quant.py`, 2 matches

Searched for regex `class.*Linear|def.*forward` (`**/python/sglang/srt/layers/quantization/**`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/modelopt_utils.py)

Searched for regex `ModelOpt.*Fp4|nvfp4|FP4` (`**/python/sglang/srt/layers/quantization/modelopt_quant.py`), 50 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1050-1050), lines 1050 to 1150

Searched for regex `quantization|fp4|nvfp4` (`**/python/sglang/srt/server_args.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1100-1100), lines 1100 to 1300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#100-100), lines 100 to 500

Searched for regex `quant_method|quant_algo|NVFP4|FP4` (`**/python/sglang/srt/model_loader/weight_utils.py`), 16 results

Searched for regex `quant|linear|forward` (`**/python/sglang/srt/models/minicpm.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/petit.py#1-1), lines 1 to 200

Searched for text `gptq_marlin fp4 marlin` (`**/python/sglang/srt/layers/quantization/gptq.py`), no results

Searched for regex `class|def.*forward` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 26 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm)

Searched for text `quant function` (`**/sgl-kernel/csrc/gemm/nvfp4*.cu`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Searched for text `fp4 marlin` (`**/sgl-kernel/csrc/gemm/marlin/**`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant_entry.cu)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_entry.cu)



Searched for text `__init__ quantization register` (`**/python/sglang/srt/layers/quantization`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 150

Fetched https://soar.openbmb.cn/toolkit

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/nvfp4_quantization_research.md)

Excellent research. Now let me also investigate how ModelOpt works and whether it can be installed on the competition environment. Let me check a few things: 

Research ModelOpt NVFP4 quantization

Searched for files matching `**/modelopt_quant.py`, 2 matches

Searched for regex `nvfp4|NVFP4|fp4_quant`, 50 results

Searched for files matching `**/nvfp4*`, 7 matches

Searched for files matching `**/gptqmodel*`, 1 match

Searched for text `modelopt`, 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1-1), lines 1 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/petit.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant_kernels.cu#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#1-1), lines 1 to 150

Searched for text `class ModelOptFp4Config`, 2 results

Searched for text `class ModelOptFp4LinearMethod`, 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#300-300), lines 300 to 800

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#858-858), lines 858 to 1050

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1053-1053), lines 1053 to 1300

Searched for text `fp4_quant|nvfp4_quant|quantize` (`**/benchmark/soar/demo_sala/*.py`), 14 results

Searched for text `fp4_quant|nvfp4_quant|quantize` (`**/scripts/**/*.py`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant.cuh)



Searched for text `nvidia_modelopt|modelopt_quant_run|quantize_to_fp4`, 5 results

Fetched https://github.com/NVIDIA/TensorRT-Model-Optimizer

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/modelopt_utils.py)

Searched for text `from_config weight_loader checkpoint`, no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/tests/test_fp4_quantize.py#1-1), lines 1 to 150

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/nvfp4_comprehensive_research.md)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_01HLuhCZnyEhnxv1zFHjUxAx__vscode-1776205772494/content.txt#1-1), lines 1 to 200

This is very valuable. Let me now read the EAGLE3 session memory to understand the Phase 2 starting point:     

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_implementation_plan.md)

Compacted conversation

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/champion_blog_analysis.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Let me pick up where we left off. Let me check the current state of both worktrees and start planning the concurrent work. 

Ran terminal command:  cd /home/oldzhu/sglang && git worktree list

Good, all three worktrees are in place. Let me check the NVFP4 inference support code and the EAGLE3 draft model to plan implementation: 

Starting: *NVFP4: Quantization pipeline research* (2/6)

NVFP4 quantization pipeline deep research

Searched for files matching `**/modelopt_quant.py`, 2 matches

Searched for files matching `**/nvfp4_quant_kernels.cu`, 1 match

Searched for files matching `**/nvfp4_scaled_mm_kernels.cu`, 1 match

Searched for files matching `**/prepare_model.sh`, 1 match

Searched for files matching `**/preprocess_model.py`, 1 match

Searched for files matching `**/gptqmodel_minicpm_sala.py`, 1 match

Searched for regex `modelopt|nvidia-modelopt|nvidia_modelopt|ModelOptFp4`, 50 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant_kernels.cu)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1-1), lines 1 to 999

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#1-1), lines 1 to 500

Searched for regex `scaled_fp4_quant|nvfp4_gemm|cutlass_scaled_fp4`, 30 results

Searched for files matching `**/nvfp4*.cu`, 6 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#800-800), lines 800 to 1500

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu#500-500), lines 500 to 688

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Searched for regex `nvidia-modelopt|nvidia_modelopt|from modelopt`, 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#200-200), lines 200 to 500

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant_entry.cu)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_scaled_mm_entry.cu)

Read [](file:///home/oldzhu/sglang/sgl-kernel/include/sgl_kernel_ops.h#200-200), lines 200 to 250

Searched for regex `CVT_FP4|ELTS_PER_THREAD|fp32_vec_to_e2m1|PackedVec` (`**/nvfp4_quant.cuh`), 16 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#500-500), lines 500 to 800

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant.cuh#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/nvfp4_quant.cuh#100-100), lines 100 to 183

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#800-800), lines 800 to 1096

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/nvfp4_complete_pipeline_research.md)

EAGLE3 training infrastructure research

Read [](file:///home/oldzhu)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models)

Searched for files matching `**/eagle*`, 20 matches

Read [](file:///home/oldzhu/sglang-eagle3)

Searched for regex `minicpm_eagle3|set_eagle3_layers_to_capture|get_eagle3_hidden_states`, 20 results

Searched for files matching `**/*train*eagle*`, no matches

Searched for text `get_eagle3_hidden_states`, no results

Searched for files matching `**/llama_eagle3.py`, 2 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#1-1), lines 1 to 100

Searched for text `class MiniCPM3Eagle3DraftModel`, no results

Searched for text `def set_eagle3_layers_to_capture`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama_eagle3.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang-eagle3/examples)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama_eagle3.py#150-150), lines 150 to 277

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama.py#200-200), lines 200 to 760

Searched for text `set_eagle3_layers_to_capture` (`sglang` · `python/sglang/srt/models/llama.py`), 1 result

Searched for files matching `/home/oldzhu/sglang-eagle3/**/train*`, no matches

Searched for regex `train|Train|TRAIN` (`/home/oldzhu/sglang-eagle3`), no results

Read [](file:///home/oldzhu/sglang-eagle3/benchmark)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama.py#768-768), lines 768 to 792

Searched codebase for "MiniCPM-SALA model config hidden_size num_layers intermediate_size", 11 results

Searched for files matching `/home/oldzhu/sglang-eagle3/examples/usage/**`, no matches

Searched for files matching `/home/oldzhu/sglang-eagle3/scripts/**`, no matches

Searched for text `class.*Eagle` (`sglang` · `python/sglang/srt/speculative`), no results

Searched for files matching `sglang` · `benchmark/**/*minicpm*`, 12 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama_eagle3.py)

Searched for text `def forward` (`sglang` · `python/sglang/srt/models/minicpm3.py`), 5 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Searched for regex `set_eagle3|get_eagle3|layers_to_capture` (`sglang` · `python/sglang/srt/models/minicpm3.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#400-400), lines 400 to 518

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#250-250), lines 250 to 390

Searched for files matching `**/*train*`, 11 matches

Searched for regex `def train|class.*Trainer|torch.optim` (`sglang`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_training_infrastructure_research.md)

Now let me check the latest leaderboard and competition rules before presenting the proposals: 



Fetched https://soar.openbmb.cn/leaderboard

Good, leaderboard is stable. team-beta #18 at 51.08, gap to #5 is ~41%. Let me now update the leaderboard memory and read key files for the proposals: 

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Now let me read the EAGLE3 draft model and the GPTQ quantization script to ensure accuracy in the proposals: 

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/models/minicpm_eagle3.py)

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/models/minicpm_eagle3.py)

Read [](file:///home/oldzhu/sglang-eagle3/python/sglang/srt/models/minicpm_eagle3.py)

Starting: *Present dual optimization proposals* (1/5)

Now let me also quickly check the MiniCPM-SALA model config to confirm dimensions: 

Searched for regex `hidden_size.*4096|num_hidden_layers.*32|intermediate_size` (`**/minicpm3.py`), 10 results

Let me check the modelopt_quant.py to understand the exact checkpoint format expected:  

Read [](file:///home/oldzhu/sglang-nvfp4/python/sglang/srt/layers/quantization/modelopt_quant.py)

Read [](file:///home/oldzhu/sglang-nvfp4/python/sglang/srt/layers/quantization/modelopt_quant.py)

Searched for regex `class ModelOptFp4Config` (`**/modelopt_quant.py`), 2 results

Read [](file:///home/oldzhu/sglang-nvfp4/python/sglang/srt/layers/quantization/modelopt_quant.py)

Read [](file:///home/oldzhu/sglang-nvfp4/python/sglang/srt/layers/quantization/modelopt_quant.py)

Read [](file:///home/oldzhu/sglang-nvfp4/python/sglang/srt/layers/quantization/modelopt_quant.py)

Now I have comprehensive understanding of both tracks. Let me present the dual optimization proposals. 

Completed: *Present dual optimization proposals* (1/5)

---

## Dual Optimization Proposals

### Current Status
- Leaderboard: team-beta #18 (51.08), gap to #5 is 41%
- Worktrees ready: sglang-eagle3 (eagle3-spec-decode), sglang-nvfp4 (nvfp4-weight-quant)
- CHANGE_0085 packaging: fcloud is shut down, needs restart to package

---

### CHANGE_0095: NVFP4 Weight Quantization

**Objective**: Replace GPTQ W4A16 (Marlin INT4→FP16 dequant + FP16 GEMM) with NVFP4 W4A4 (native Blackwell FP4×FP4 tensor core GEMM). Champion Slightwind uses NVFP4 and scores 79.22.

**Expected Gain**: 
- At high batch sizes (Smax): FP4 tensor cores offer ~2x compute throughput over FP16 → potential 15-30% Smax improvement
- At low batch sizes (S1): memory-bound, gain may be modest (0-10%) since weight size is same (4-bit either way)
- Activation quantization overhead partially offsets gains at small M

**Rule Compliance**:
- On-site quantization ✓ (custom Python script, ~10-20 min)
- ≤ 2GB ✓ (FP4 model is same or slightly smaller than GPTQ INT4)
- No forbidden tricks ✓
- SM120 (compute 12.0) hardware confirmed ✓

**Risk to Accuracy**: MEDIUM
- FP4 E2M1 has only 16 distinct values (vs GPTQ's 16 INT4 values) — similar information capacity
- Per-block scales (group_size=128) vs GPTQ's group_size=128 — comparable granularity
- BUT: activation quantization to FP4 online introduces additional error not present in GPTQ W4A16
- Mitigation: calibrate `input_scale` from perf_public_set.jsonl forward passes

**Files to Create/Change** (in `sglang-nvfp4` worktree):
1. **NEW**: `benchmark/soar/demo_sala/quantize_nvfp4.py` — Offline FP4 weight quantization script:
   - Loads BF16 model from `/root/models/openbmb/MiniCPM-SALA-Copy`
   - For each linear layer: compute per-block scales, quantize weights to E2M1, pack uint8
   - Run calibration forward passes to compute input_scale per layer
   - Save checkpoint with `quant_algo: "NVFP4"` config
2. **MODIFY**: prepare_model.sh — call `quantize_nvfp4.py` instead of `preprocess_model.py`
3. **MODIFY**: prepare_env.sh — change `--quantization gptq_marlin` → `--quantization modelopt_fp4`, remove `--force-dense-minicpm` (may not be needed with NVFP4)

**Test Commands**:
```bash
# Quantize model on fcloud
python3 quantize_nvfp4.py --input /root/models/openbmb/MiniCPM-SALA-Copy --output /root/models/nvfp4_minicpm

# Start server with NVFP4
python3 -m sglang.launch_server --model-path /root/models/nvfp4_minicpm --quantization modelopt_fp4 ...

# Accuracy + speed eval
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

**Success/Failure**:
- Success: accuracy ≥ 79% (normalized ≥ 99%), speed ≤ Smax ≤ 30s
- Failure: accuracy < 77% or speed regression vs GPTQ baseline

---

### CHANGE_0100: EAGLE3 Phase 2 — Draft Model Training

**Objective**: Train the EAGLE3 draft head for MiniCPM-SALA to enable speculative decoding. This is the single highest-impact optimization vector — top 2 teams (#1 FlashSALA 94.29, #2 智算一队 90.22) likely use speculative decoding.

**Expected Gain**: 
- 2-4x decode speedup (at 60-80% acceptance rate)
- S1: 113s → ~40-60s, S8: 41s → ~15-25s, Smax: 34s → ~15-20s
- If successful, could jump to top 5-8 range

**Rule Compliance**:
- Draft model weights ≤ ~600MB FP16 (within 2GB total) ✓
- Speculative heads explicitly allowed by rules ✓
- Apache 2.0 compatible ✓

**Risk to Accuracy**: LOW
- Speculative decoding is mathematically exact (rejection sampling guarantees same distribution)
- Only risk: implementation bugs causing wrong acceptance/rejection

**Implementation Plan**:
1. **NEW**: `benchmark/soar/demo_sala/train_eagle3_draft.py` — Training script:
   - Load MiniCPM-SALA BF16 target model using HuggingFace `transformers`
   - Forward pass through calibration data, capture aux_hidden_states at layers [2, 16, 29]
   - Create draft model: FC(3×4096→4096) + 1 decoder layer (~58M params)
   - Train with KL divergence loss: `KL(target_logits || draft_logits)`
   - Optimizer: AdamW, lr=1e-4, ~1000 steps, batch_size=1-2
   - Training data: `perf_public_set.jsonl` (150 samples) — small but should suffice for simple head
   - Save: model.safetensors + config.json with `architectures: ["MiniCPMForCausalLMEagle3"]`

2. **Config template** (`config.json` for draft model):
```json
{
  "architectures": ["MiniCPMForCausalLMEagle3"],
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_hidden_layers": 1,
  "vocab_size": 150528,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-5,
  "eagle_config": {
    "use_aux_hidden_state": true,
    "eagle_aux_hidden_state_layer_ids": [2, 16, 29]
  },
  "draft_vocab_size": null,
  "tie_word_embeddings": false
}
```

3. **MODIFY** prepare_env.sh to add: `--speculative-algorithm EAGLE3 --speculative-draft-model-path /path --speculative-num-steps 5 --speculative-num-draft-tokens 64 --speculative-eagle-topk 1`

**Test Commands**:
```bash
# Train on fcloud (using BF16 model + A800 GPU)
cd /root/sglang-minicpm/benchmark/soar/demo_sala
python3 train_eagle3_draft.py --model-path /root/models/openbmb/MiniCPM-SALA-Copy --output-path /root/models/eagle3_draft --data-path /root/data/perf_public_set.jsonl --num-steps 1000

# Test with spec decode
python3 scripts/fcloud/fcloud_workflow.py restart-server  # with EAGLE3 args
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

**Success/Failure**:
- Success: acceptance rate ≥ 60%, S1 decrease ≥ 30%, accuracy maintained
- Failure: acceptance rate < 40% or speed regression

---

### Recommended Priority

1. **EAGLE3 training script first** — it's the only path that can close the 41% gap to top 5. NVFP4 alone (~15-30% gain) cannot.
2. **NVFP4 in parallel** — can be developed independently and tested separately
3. **Both merge to main** only after proven on fcloud

Shall I proceed with implementing both? If so, I'll start with the EAGLE3 training script in sglang-eagle3, then the NVFP4 quantization script in sglang-nvfp4.

