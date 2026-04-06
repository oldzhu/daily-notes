

The revert is already pushed (commit `feb29c4bc`). You can pull and test now.

Let me investigate all three paths in parallel. 

Created 3 todos

Path A: sparse attention crash root cause

Searched for text `cuda_graph` (`**/benchmark/soar/demo_sala/sglang/**`), 20 results

Searched for regex `fp8|kv_cache_dtype` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_backend.py`), 15 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/common.py#1-1), lines 1 to 200

Searched for text `cuda_graph` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#200-200), lines 200 to 500

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#200-200), lines 200 to 500

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#500-500), lines 500 to 800

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#800-800), lines 800 to 1100

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1100-1100), lines 1100 to 1400

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1400-1400), lines 1400 to 1600

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1600-1600), lines 1600 to 1900

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#500-500), lines 500 to 800

Searched for regex `flashinfer|cuda_graph|decode_wrapper|decode_workspace` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_attention_kernels.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#800-800), lines 800 to 1200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#1-1), lines 1 to 100

Searched for regex `sparse_cache_seqlens|cu_seqlens_k1|cu_seqlens_k2|history_compress_k1|compress_k1` (`**/benchmark/soar/demo_sala/sglang/**/forward_batch_info.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#100-100), lines 100 to 350

Searched for text `convert_sparse_page_table_to_flashinfer` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_sparse_kernels.py`), 1 result

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_attention_kernels.py#330-330), lines 330 to 500

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#675-675), lines 675 to 714

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#200-200), lines 200 to 675

Searched for text `build_decode_topk_metadata` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_sparse_utils.py`), 1 result

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1200-1200), lines 1200 to 1600

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1618-1618), lines 1618 to 1663

Searched for files matching `**/sparse_kernel_extension*`, no matches

Searched for text `get_block_table_v3` (`**/benchmark/soar/demo_sala/sglang/**`), 1 result

Searched for files matching `**/CHANGE_003*.md`, 20 matches

Searched for text `sparse_kernel_extension` (`**/benchmark/soar/demo_sala/**`), 3 results

Searched for files matching `**/CHANGE_0039*.md`, 2 matches

Searched for files matching `**/CHANGE_0040*.md`, 2 matches

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0039_minicpm_fp8_sparse_compression_scratch_path.en.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0040_minicpm_sparse_decode_slice_fix.en.md#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0038_minicpm_fp8_sparse_topk_bf16_bridge.zh.md#1-1), lines 1 to 80

Searched for text `scratch_only` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_sparse_kernels.py`), no results

Searched for regex `bfloat16|bf16|scratch_only` (`**/benchmark/soar/demo_sala/sglang/**/minicpm_backend.py`), 8 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#800-800), lines 800 to 870

Searched for text `write_sparse_k1` (`**/benchmark/soar/demo_sala/sglang/**/common.py`), 3 results

Searched for regex `sparse_cache_seqlens|cu_seqlens_k1|history_compress_k1` (`**/benchmark/soar/demo_sala/sglang/**/schedule_batch.py`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/common.py#520-520), lines 520 to 580

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1160-1160), lines 1160 to 1200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/managers/schedule_batch.py#2040-2040), lines 2040 to 2060

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/managers/schedule_batch.py#2060-2060), lines 2060 to 2120

Path B: torch.compile feasibility

Searched for text `torch_compile` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/server_args.py`), 6 results

Searched for regex `torch_compile|torch\.compile` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/**`), 80 results

Searched for regex `torch_compile|compile` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/**`), 20 results

Searched for regex `piecewise_cuda_graph|piecewise-cuda-graph` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/**`), 40 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#130-130), lines 130 to 220

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/model_executor/model_runner.py#1495-1495), lines 1495 to 1520

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py#510-510), lines 510 to 540

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#100-100), lines 100 to 500

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py#1060-1060), lines 1060 to 1100

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#230-230), lines 230 to 350

Searched for regex `torch\.compile|graph_break|custom_op|torch\.no_grad` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), no results

Searched for regex `torch\.compile|graph_break|custom_op` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#500-500), lines 500 to 600

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#600-600), lines 600 to 673

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py#380-380), lines 380 to 450

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 100

Searched for regex `GPTQMarlinLinearMethod|marlin_gemm|register_fake` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py`), 6 results

Searched for regex `class SimpleGLAAttnBackend` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 1 result

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py#560-560), lines 560 to 750

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1560

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/model_executor/model_runner.py#2000-2000), lines 2000 to 2060

Searched for text `marlin_gemm` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py`), no results

Searched for regex `register_fake|torch\.library` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/marlin_utils.py`), no results

Searched for text `def apply` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py`), 3 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/gptq.py#787-787), lines 787 to 870

Searched for regex `apply_gptq_marlin_linear|register_fake|marlin_gemm|dynamo` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/marlin_utils.py`), 17 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#460-460), lines 460 to 530

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#860-860), lines 860 to 970

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#400-400), lines 400 to 450

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1560-1560), lines 1560 to 1652

Path C: kernel optimization opportunities

Read [](file:///home/oldzhu/sglang/3rdparty/infllmv2_cuda_impl)

Read [](file:///home/oldzhu/sglang/3rdparty/sparse_kernel)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/configs/minicpm.py)

Searched for text `fused` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py`), no results

Searched for text `fused_qk_norm_rope` (`**/benchmark/soar/demo_sala/**`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#150-150), lines 150 to 400

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 200

Searched for regex `fuse_topk|split_stage1|fuse.topk|split.stage` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py`), no results

Searched for regex `fused_add_rms_norm|residual.*rms|rms.*residual` (`**/sgl-kernel/src/**`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#400-400), lines 400 to 600

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#200-200), lines 200 to 400

Searched for regex `fuse.topk|split.stage1|fuse_topk|split_stage` (`**/benchmark/soar/demo_sala/**`), 20 results

Searched for regex `fused_add_rms|rms_norm` (`**/sgl-kernel/src/**`), no results

Searched for regex `fused_add_rms_norm` (`**/sgl-kernel/**`), 6 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#600-600), lines 600 to 673

Searched for regex `fused_add_rms_norm|class RMSNorm` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/layernorm.py`), 5 results

Searched for regex `split_stage1|fuse_topk` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`), 19 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#110-110), lines 110 to 160

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#240-240), lines 240 to 300

Searched for regex `scale_depth|scale_emb|residual|hidden_states =` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py`), 20 results

Searched for text `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO` (`**/benchmark/soar/demo_sala/sglang/python/sglang/**`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#600-600), lines 600 to 700

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/layernorm.py#60-60), lines 60 to 200

Searched for text `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO` (`**/benchmark/soar/demo_sala/sglang/python/sglang/**`), no results

Searched for text `LIGHTNING_FAST_OUTPUT_GATE` (`**/benchmark/soar/demo_sala/sglang/python/sglang/**`), no results

Searched for text `RECURRENT_THRESHOLD` (`**/benchmark/soar/demo_sala/sglang/python/sglang/**`), no results

Searched for files matching `**/benchmark/soar/demo_sala/**/config.json`, no matches

Searched for text `LIGHTNING_FAST_STATE_IO` (`**/benchmark/soar/demo_sala/**`), 4 results

Searched for text `LIGHTNING_FAST_OUTPUT` (`**/benchmark/soar/demo_sala/**`), 4 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#400-400), lines 400 to 900

Ran terminal command:  wc -l /home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py

Searched for regex `class SimpleGLAAttnBackend` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 1 result

Searched for regex `fused_qk_norm_rope` (`**/sgl-kernel/**`), 12 results

Searched for regex `enable_fused_qk_norm_rope|fused_qk_norm_rope` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1600

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu#360-360), lines 360 to 420

Searched for regex `class.*Backend|def forward_decode|def forward_extend|def sparse_get_topk` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`), 5 results

Searched for regex `enable_fused_qk_norm_rope|fused_qk_norm_rope|from sgl_kernel` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py`), no results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1600-1600), lines 1600 to 1651

Searched for regex `class SiluAndMul` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/activation.py`), 1 result

Searched for regex `silu_and_mul|SiluAndMul` (`**/sgl-kernel/**`), 20 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#180-180), lines 180 to 210

Searched for text `mixer_types` (`**/benchmark/soar/demo_sala/**`), 20 results

Searched for regex `force.dense.minicpm|force_dense_minicpm` (`**/benchmark/soar/demo_sala/**`), 11 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#415-415), lines 415 to 440

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/server_args.py#1510-1510), lines 1510 to 1540

Searched for regex `def forward_native|class.*RotaryEmb` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/rotary_embedding.py`), 14 results

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/perf_args_survey.md)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/rotary_embedding.py#248-248), lines 248 to 310

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/rotary_embedding.py#190-190), lines 190 to 248

Searched for regex `def forward_cuda` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/rotary_embedding.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/rotary_embedding.py#342-342), lines 342 to 410

Ran terminal command:  find /home/oldzhu/sglang/benchmark/soar/demo_sala -name "*.json" -path "*/config*" 2>/dev/null | head -5

Searched for regex `minicpm4|lightning|mixer_types` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 6 results

Searched for regex `dense_as_sparse` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/server_args.py`), 1 result

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/server_args.py#1518-1518), lines 1518 to 1530

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/server_args.py#530-530), lines 530 to 545

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/model_executor/model_runner.py#1485-1485), lines 1485 to 1510

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/kernel_optimization_survey.md)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_bdrk_01JpXkTi1Edu7KzaCGMhs7Nx__vscode-1775430475736/content.txt#1-1), lines 1 to 200

Completed: *Path A: Investigate sparse attn crash* (1/4)

Starting: *Propose priority-ranked next steps* (4/4)

Now let me verify a critical finding before presenting the proposal. I need to check whether CHANGE_0038-0040 patches are actually applied in the current codebase: 

Searched for regex `bfloat16|bf16.*bridge|scratch_only|fp8.*cast` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`), 6 results

Searched for text `slice(` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/mem_cache/common.py`), 6 results

Good — the CHANGE_0040 slice fix IS applied in this codebase. Let me check if CHANGE_0038 BF16 bridge is applied:  

Searched for regex `kv_cache_dtype|fp8` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`), 15 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#830-830), lines 830 to 890

Let me check near the `sparse_get_topk_impl` function for CHANGE_0038 BF16 bridge:  

Searched for text `sparse_get_topk_impl` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`), 5 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#795-795), lines 795 to 830

No BF16 bridge in `sparse_get_topk_impl`. The subagent analysis was correct — CHANGE_0038 code changes are NOT applied (only the doc exists). Let me check CHANGE_0039: 

Searched for text `scratch_only` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`), no results

Confirmed: both CHANGE_0038 and CHANGE_0039 code patches are NOT in the codebase (only docs exist). CHANGE_0040 slice fix IS applied. Now let me check the model forward for Path C: 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py#475-475), lines 475 to 510

Good, I have all the data. Let me now check what `sgl_kernel` already offers for scaled residual:   

Searched for regex `fused_add_rmsnorm|scaled.*rmsnorm|rmsnorm.*scale` (`**/benchmark/soar/demo_sala/sglang/**"`), no results

Searched for regex `fused_add_rmsnorm` (`**/benchmark/soar/demo_sala/sglang/**"`), no results

Searched for text `fused_add_rmsnorm`, 20 results

Let me check the existing `fused_add_rmsnorm` signature to see if it supports a scale factor:  

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/elementwise.py#49-49), lines 49 to 100

The existing `fused_add_rmsnorm` does `residual += input` then `input = rmsnorm(residual)`. MiniCPM needs `residual = residual + input * scale` then `output = rmsnorm(residual)`. The scale factor is the key difference. This would need a new kernel variant. 

Completed: *Propose priority-ranked next steps* (4/4)

Here's the comprehensive analysis and priority-ranked plan:

---

## Investigation Results

### Path A: Fix Sparse Attention (FP8 + CUDA Graph)
**Status**: HIGH POTENTIAL, 5 bugs identified

| Bug | Status | Severity | Issue |
|---|---|---|---|
| 1. FP8 query → sparse scorer | **NOT FIXED** (CHANGE_0038 doc only) | CRITICAL | `torch.bmm(fp8_q, bf16_k)` dtype mismatch |
| 2. Tuple indexing in k1/k2 writes | **FIXED** (CHANGE_0040 applied) | CRITICAL | slice fix is in place |
| 3. Compress-K mutates FP8 cache | **NOT FIXED** (CHANGE_0039 doc only) | MODERATE | FP8 in-place write → lossy readback |
| 4. **NEW**: kv_indptr overwrite inside CUDA graph ≠ begin_forward plan | **NOT FIXED** | **CRITICAL** | `begin_forward` plans with static indptr `[0,K,2K,...]` but decode rewrites it with variable `sparse_cache_seqlens < K`. FlashInfer workspace OOB → crash |
| 5. FlashInfer FP8 + num_kv_heads=1 edge case | **UNKNOWN** | MEDIUM | Under-tested config |

**Bug 4 is the likely root cause of why crash persisted after CHANGE_0038-0040.** The other fixes addressed scorer/metadata bugs, but the FlashInfer plan mismatch would still cause the CUDA graph replay to crash.

**Expected gain if fixed**: 20-40% on long-context prefill/decode (sparse attention computes on topk blocks instead of full sequence).

### Path B: torch.compile
**Status**: NOT WORTH PURSUING

- Lightning attention layers have data-dependent control flow → graph breaks
- `fla` Triton kernels likely lack `register_fake` implementations
- Expected gain only 2-8% on decode even if it worked
- Compilation adds 5-10min startup

### Path C: Kernel Fusion Optimizations
**Status**: MODERATE POTENTIAL, 3 opportunities

| Optimization | Gain | Effort | 
|---|---|---|
| **Fused scaled-residual-add + RMSNorm** | 3-8% | HIGH (new CUDA kernel) |
| **Remove float32 RoPE cast** | 2-4% | LOW (code change only) |
| **Integrate fused_qk_norm_rope for lightning layers** | 2-5% | MEDIUM |

The existing `sgl_kernel.fused_add_rmsnorm` does `residual += input` but MiniCPM needs `residual = residual + input * scale`. A new kernel variant would be needed.

---

## Recommended Priority Order

| Priority | What | Expected gain | Risk |
|---|---|---|---|
| **1st** | **Path C.2: Remove float32 RoPE cast** | 2-4% | LOW |
| **2nd** | **Path A: Fix sparse attention** (all 4 bugs) | 20-40% | HIGH |
| **3rd** | **Path C.1: Fused scaled-residual-add + RMSNorm** | 3-8% | MEDIUM |

**Rationale**: Path C.2 (remove fp32 RoPE cast) is the lowest-risk, easiest-to-implement change. We apply and test it first. Then tackle Path A which is the huge win but carries crash risk.

---

Shall I start with **Path C.2 (remove float32 RoPE cast in standard attention layers)**? It's a small code change in minicpm.py — just removing the `.float()` / `.to(orig_dtype)` casts and letting the CUDA RoPE kernel operate directly on bf16.
