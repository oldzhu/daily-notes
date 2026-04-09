I would like to have 3 subagents to do throughly research and plan in the 3 paths concurrently and provide plan/resarch document for our discussion and review how to move forward - pick 1 or 2 of 3 or do 3 1 or 1 or concurrently.
1. for path 1 - focus on benchmark speed deep optimization without accuracy loss e.g  code review in the code path for  path 1 to see any chance to improving speed? or refer the skills and way at https://github.com/gxinlong/cuda-optimization-skill  using NCU profiling，生成 → 验证 → 分析 → 优化的闭环循环?
2. for 2, the same as above  focus on benchmark speed deep optimization without accuracy loss, specially on 可选路径：EAGLE3 多层 Draft Head（需算法创新）mentioned at https://soar.openbmb.cn/toolkit
3. for 3, do through research and investigation, understand why understand why GPTQ breaks the sparse scoring and possible solution.
========
Let me first fetch the referenced URLs to provide quality context to the subagents, then launch all 3 research agents concurrently. 

Created 3 todos



Fetched 2 resources

Starting: *Launch 3 research subagents* (2/3)

Now launching all 3 research subagents concurrently:  

Path 1: Dense+GPTQ+FP8 speed optimization

Searched for files matching `**/python/sglang/srt/models/minicpm_sala.py`, no matches

Searched for files matching `**/python/sglang/srt/layers/attention/minicpm_backend.py`, 2 matches

Searched for files matching `**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, 2 matches

Searched for files matching `**/python/sglang/srt/layers/quantization/gptq.py`, 2 matches

Searched for files matching `**/python/sglang/srt/layers/quantization/kv_cache.py`, 2 matches

Searched for files matching `**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`, 1 match

Searched for files matching `**/python/sglang/srt/models/*minicpm*.py`, 8 matches

Searched for regex `minicpm_sala|MiniCPMSALA` (`**/python/sglang/srt/models/**`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#800-800), lines 800 to 822

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 120

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/kv_cache.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 100

Searched for regex `def forward|force_dense|force_layer|GLA|attention` (`**/python/sglang/srt/models/minicpm.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#200-200), lines 200 to 400

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#400-400), lines 400 to 600

Searched for regex `force.*dense|force_dense` (`**/python/sglang/srt/**`), 20 results

Searched for regex `class.*Backend|def forward|force_dense` (`**/python/sglang/srt/layers/attention/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1450-1450), lines 1450 to 1550

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1550-1550), lines 1550 to 1600

Searched for regex `def forward|apply_gptq_marlin|GPTQ` (`**/python/sglang/srt/layers/quantization/gptq.py`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 100

Searched for regex `apply_gptq_marlin_linear|def apply|forward` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#200-200), lines 200 to 350

Searched for regex `max_running_requests|mem_fraction_static|schedule_conservativeness|chunked_prefill` (`**/python/sglang/srt/server_args.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#464-464), lines 464 to 535

Searched for regex `chunked_prefill_size.*=|max_running_requests.*=|mem_fraction_static.*=|schedule_conservativeness.*=` (`**/python/sglang/srt/server_args.py`), 20 results

Searched for regex `cuda_graph|forward_batch|synchronize|profile` (`**/python/sglang/srt/model_executor/**`), 20 results

Searched codebase for "RoPE rope rotation embeddings MiniCPM", no results

Searched for files matching `**/prepare_model.sh`, 1 match

Searched for files matching `**/preprocess_model.py`, 1 match

Searched for regex `forward|decode|extend|profile` (`**/python/sglang/srt/layers/radix_attention.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/benchmark/soar)

Searched for regex `forward|cuda_graph|profile|attention_backend` (`**/python/sglang/srt/model_executor/model_runner.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/radix_attention.py#1-1), lines 1 to 50

Searched for regex `capture|replay|forward` (`**/python/sglang/srt/model_executor/cuda_graph_runner.py`), 20 results

Read memory

Path 2: EAGLE3 speculative decoding research

Searched for files matching `**/speculative/*.py`, 20 matches

Searched for files matching `**/models/minicpm*.py`, 8 matches

Searched for files matching `**/layers/attention/*.py`, 20 matches

Searched for files matching `**/models/llama*.py`, 14 matches

Searched for regex `minicpm.*sala|SALA`, 20 results

Searched for files matching `**/hybrid_linear_attn*.py`, 2 matches

Searched for files matching `**/radix_linear*.py`, no matches

Read memory

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/llama_eagle3.py#1-1), lines 1 to 150

Searched for regex `multilayer|EAGLE3` (`**/speculative/*.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#200-200), lines 200 to 750

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_utils.py#1-1), lines 1 to 100

Searched for regex `mixer_types|lightning` (`**/srt/models/minicpm.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_utils.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_info.py#1-1), lines 1 to 150

Searched for regex `is_target_verify`, 20 results

Searched for files matching `**/config/json`, no matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#100-100), lines 100 to 300

Searched for regex `SALA|mixer_types.*lightning`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#250-250), lines 250 to 990

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/eagle_worker.py#1-1), lines 1 to 150



Searched for regex `SimpleGLAAttnBackend|hybrid_linear`, 15 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#100-100), lines 100 to 800

Fetched https://soar.openbmb.cn/toolkit

Searched for regex `draft_token_num|num_draft_tokens` (`**/speculative/*.py`), 10 results

Searched for regex `lookahead|prompt_lookup` (`**/speculative/*.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py#1-1), lines 1 to 100

Path 3: GPTQ breaks sparse investigation

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 100

Searched for regex `sparse_config|topk|kernel_size|kernel_stride` (`**/python/sglang/srt/layers/attention/minicpm_*.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#100-100), lines 100 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#50-50), lines 50 to 250

Searched for regex `fused_attn_pooling_online_topk|topk|score` (`**/python/sglang/srt/layers/attention/minicpm_fuse_kernel.py`), 50 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#300-300), lines 300 to 800

Searched for regex `def forward.*attention|qkv_proj|q_proj|k_proj` (`**/python/sglang/srt/layers/attention/*.py`), 20 results

Searched for regex `def.*forward|compressed_attention|infllmv2` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 19 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#800-800), lines 800 to 850

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1050-1050), lines 1050 to 1100

Searched for regex `def forward\(|query_layer|proj_fn` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 9 results

Searched for regex `class.*Attention|def forward.*query` (`**/python/sglang/srt/layers/*.py`), 6 results

Searched for regex `def forward|q_proj|k_proj|qkv` (`**/python/sglang/srt/layers/radix_attention.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/radix_attention.py)

Searched codebase for "MiniCPM attention projection Q K V GPTQ quantization", no results

Searched for regex `q_proj|k_proj|qkv` (`**/python/sglang/srt/model_executor/prefill_batch_stages/**`), no results

Searched for regex `q_proj|q_gate|attention` (`**/python/sglang/srt/models/*.py`), 20 results

Searched for files matching `**/python/sglang/srt/models/*minicpm*`, 8 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#170-170), lines 170 to 280

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#200-200), lines 200 to 280

Searched for regex `GPTQ|dequant|forward` (`**/python/sglang/srt/layers/quantization/*.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization)

Searched for regex `def forward|class` (`**/python/sglang/srt/layers/quantization/gptq.py`), 20 results

Searched for regex `forward|apply_weight` (`**/python/sglang/srt/layers/quantization/gptq.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#400-400), lines 400 to 500

Searched for regex `class.*Linear.*Method|def apply_weight|def execute` (`**/python/sglang/srt/layers/quantization/gptq.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#500-500), lines 500 to 600

Searched for regex `def gptq_gemm|from.*gptq_gemm|import.*gptq_gemm` (`**/python/sglang/srt/layers/quantization/gptq.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 50

Searched for regex `CHANGE_0071|sparse|GPTQ|FP8` (`**/docs/soar_2026_changes/*.md`), 20 results

Read memory

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0071_minicpm_sparse_triton_int64_guard.zh.md#1-1), lines 1 to 100

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/sparse_attention_bugs.md)

Searched for regex `k_scale|Bug 3` (`**/python/sglang/srt/layers/attention/minicpm_sparse_utils.py`), 10 results

Searched for regex `k_scale|descale|Bug 3` (`**/docs/soar_2026_changes/*.md`), 12 results

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/gptq_sparse_research.md)

All 3 subagents completed. Let me consolidate their findings:  

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Bf12Bw1q1WxZqJwsbJEeRJ__vscode-1775603769566/content.txt#1-1), lines 1 to 200

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Bf12Bw1q1WxZqJwsbJEeRJ__vscode-1775603769566/content.txt#200-200), lines 200 to 380

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Bf12Bw1q1WxZqJwsbJEeRJ__vscode-1775603769566/content.txt#380-380), lines 380 to 460

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Dyb9ZqYNKkJPMengZRzD1p__vscode-1775603769607/content.txt#1-1), lines 1 to 200

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Dyb9ZqYNKkJPMengZRzD1p__vscode-1775603769607/content.txt#200-200), lines 200 to 400

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Qw4d6PXpyHYNRxR5m237y1__vscode-1775603769638/content.txt#1-1), lines 1 to 200

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Qw4d6PXpyHYNRxR5m237y1__vscode-1775603769638/content.txt#200-200), lines 200 to 450

Compacted conversation

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_01Qw4d6PXpyHYNRxR5m237y1__vscode-1775603769638/content.txt#450-450), lines 450 to 604

Now I have all three research outputs fully read. Let me also quickly check the session memory for current status, then present the consolidated findings. 

Read memory

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/sparse_attention_testing_status.md)

Good, I have all the context. Now let me consolidate all three research paths into a clear comparison document for the user.

---

## Consolidated Research: 3 Optimization Paths

Here's the synthesis of all three subagent research outputs:

---

### Path 1: Dense + GPTQ + FP8 Speed Optimization

**Current baseline:** 78.84% accuracy (Test 7), using `--force-dense-minicpm`

**Key bottlenecks identified (call chain analysis):**
1. Attention backend choice: 15-25% of runtime
2. Marlin GEMM kernel config: 10-20%
3. KV cache FP8 dequant: 5-10%

**Optimization phases:**
| Phase | Action | Effort | Expected Gain |
|-------|--------|--------|---------------|
| A (Quick wins) | CUDA graph tuning, `chunked_prefill`, `max_running_requests`, `schedule_conservativeness` | 1-2h | 5-15% speed |
| B (Kernel) | KV dequant fusion, Lightning vs dense tradeoff tuning | 4-8h | 10-20% speed |
| C (Accuracy) | Recover 78.84% → ≥80% via per-layer mixed precision, extended GPTQ calibration | 8-16h | +1-2% accuracy |

**Verdict:** Low risk, medium reward. We keep GPTQ+FP8+dense and squeeze speed. Accuracy sits at 78.84% — just barely below the 80% C=0.8 threshold if relative accuracy matters. NCU profiling would be the first step.

---

### Path 2: Non-quant + Sparse + EAGLE3 Speculative Decoding

**Current baseline:** 83.02% accuracy (Test 4), non-quant + bf16 + sparse

**Critical blocker:** Lightning Attention's recurrent state computation is **incompatible with tree verification masks** needed for EAGLE3 tree-based speculation.

**Workaround options:**
| Option | Speedup | Risk | Effort |
|--------|---------|------|--------|
| A: Flat draft (no tree) | 1.3-1.5x | Low | 3-4 weeks |
| B: Branch state tracking | 1.5-2.0x | HIGH | 6-8 weeks |
| C: Skip Lightning in draft | 1.4-1.8x | Low | 4-5 weeks |
| D: Algorithmic innovation | 2.0-2.5x | Very High | 8+ weeks |

**Additional requirements:** Training draft head needs ~50K-100K examples, 12-24h training.

**Verdict:** High effort (6-9 weeks total), high risk due to Lightning Attention blocker. Best theoretical payoff (1.3-2.2x) but timeline likely exceeds competition window unless we go with flat draft (Option A).

---

### Path 3: Fix GPTQ + Sparse Incompatibility

**Root cause confirmed:** topk=8 selects only 0.4% of ~2000 candidate blocks. GPTQ introduces 1-2% per-element error → 3-5% score shift → wrong blocks selected. NIAH collapses 100%→37% because needle must be in exact top-8.

**Top solutions:**
| Solution | Effort | Expected Accuracy | Speed Impact |
|----------|--------|-------------------|--------------|
| **Increase topk (8→16)** | 1-2h | 57%→70-75% | -10-15% |
| **FP32 scoring** | 2-3h | 57%→80-85% | -5-10% |
| **Disable sparse for GPTQ (fallback to dense)** | 1h | 57%→78.84% | Dense speed |
| **topk=16 + FP32 scoring combined** | 3-4h | 57%→85%+ | -15-20% |

**Verdict:** Low effort, low risk. Increasing topk and/or using fp32 scoring could recover most of the GPTQ+sparse accuracy while keeping sparse speed benefits. The combined approach (topk=16 + fp32) might push accuracy to 85%+, which would be the best accuracy of any GPTQ config.

---

### Cross-Path Comparison

| Criterion | Path 1 (Dense Speed) | Path 2 (EAGLE3) | Path 3 (Fix GPTQ+Sparse) |
|-----------|----------------------|------------------|---------------------------|
| **Effort** | Low-Medium | Very High (6-9wk) | Very Low (1-4h) |
| **Risk** | Low | High | Low |
| **Accuracy ceiling** | ~78-80% | ~83% (C=1.0) | ~80-85% |
| **Speed ceiling** | 10-20% faster dense | 1.3-2.2x | Sparse speed (faster than dense) |
| **Timeline** | Days | Weeks-Months | Hours |
| **Dependencies** | NCU profiling | Training infra | None |

### Recommended Strategy

**Priority 1 — Path 3 first (immediate, 1-4 hours):**
Try `topk=16` and/or fp32 scoring with GPTQ+sparse. If accuracy recovers to ≥80%, we get both GPTQ compression and sparse speed benefits — the best of both worlds.

**Priority 2 — Path 1 in parallel (days):**
Tune runtime parameters and profile with NCU for speed gains regardless of which attention mode we use. These optimizations apply to any configuration.

**Priority 3 — Path 2 only if timeline allows (weeks):**
EAGLE3 is the highest theoretical payoff but the Lightning Attention blocker makes it risky. Defer unless we have significant time remaining.

**Quick win combo:** If Path 3 recovers GPTQ+sparse to ~83% accuracy, combined with Path 1 speed tuning, we could potentially reach a competitive score without EAGLE3.

---

Shall I proceed with Path 3 (increase topk / fp32 scoring) as the first optimization? I'll prepare a detailed proposal per the workflow rules.
