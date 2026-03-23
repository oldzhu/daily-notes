我会先做纯分析，不改代码。先定位 MiniCPM-SALA 里 mixed attention 层的实现路径、对应调用到的 sgl-kernel / flashinfer / torch 算子，再结合你给的参考文章整理出最值得盯的前 3 个热点。

Created 5 todos

这一批我会同时抓两类上下文：仓库里 mixed attention 的真实实现入口，以及你给的公众号文章里提到的优化思路，最后合并成可执行的优化 proposal。



Searched for regex `lightning attention|lightning_attention|full attention|mixed attention|MiniCPM|SALA` (`**/benchmark/soar/demo_sala/**`), 200 results

Searched for regex `lightning attention|lightning_attention|full attention|mixed attention|MiniCPM|SALA` (`**/python/sglang/**`), 200 results

Searched for regex `lightning attention|lightning_attention|full attention|mixed attention|MiniCPM|SALA` (`**/sgl-kernel/**`), 18 results

Fetched https://mp.weixin.qq.com/s/QwfzzGtDU7_B_mopjndMaw

已经定位到主路径不是 sgl-kernel 先直接吃满，而是 `MiniCPMHybridConfig` + minicpm_backend.py 在 sparse/full/lightning 混合路径上调不同后端。下一步我会把 mixed layer 的真实算子链拆开，避免把精力浪费在不在热路径上的 kernel。

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1-1), lines 1 to 420

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#420-420), lines 420 to 980

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#980-980), lines 980 to 1600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/minicpm.py)

现在已经能确认一半事实：full-attention 那 25% 层走的是 minicpm_backend.py 这条复杂 sparse/full 混合链，而且它里面确实有几个非常重的自定义 kernel。接下来补另一半，把 lightning 层对应的 `SimpleGLA` / linear-attention 路径和其热点算子补齐，再给你一个跨两类层的 top-3 排名。

Searched for regex `SimpleGLA|simple_gla|lightning_layer_ids|lightning_head_dim|mamba2_cache_params|gla` (`**/python/sglang/srt/**`), 200 results

Searched for regex `SimpleGLA|simple_gla|lightning_layer_ids|lightning_head_dim|mamba2_cache_params|gla` (`**/benchmark/soar/demo_sala/sglang/python/sglang/srt/**`), 200 results

Searched for files matching `sglang` · `python/sglang/srt/layers/**/*gla*`, no matches

Searched for files matching `sglang` · `python/sglang/srt/models/minicpm*.py`, 4 matches

已经确认 lightning 层不是走 sgl-kernel 目录里的独立 `lightning` 文件，而是复用 SGLang 现有 hybrid / linear-attention 基础设施。下一步我直接看模型层构造和 forward 路由，弄清每层到底是在哪个模块里分叉。

Searched for regex `mixer_types|lightning_layer_ids|full_attention_layer_ids|SimpleGLAStateShape|mamba2_cache_params|HybridLinearKVPool|MiniCPMHybrid` (`**/python/sglang/srt/models/**`), 12 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 420

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#340-340), lines 340 to 430

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#1470-1470), lines 1470 to 1535

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#420-420), lines 420 to 683

Searched for regex `class SimpleGLAAttnBackend|def forward\(|chunk_delta_h|gated|fused_recurrent|fused_chunk|flash_linear_attention|gla` (`**/python/sglang/srt/layers/attention/**`), 200 results

Searched for files matching `sglang` · `python/sglang/srt/layers/attention/fla/**`, 16 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#560-560), lines 560 to 760

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 320

Compacted conversationTracing the lightning path now. I’ve confirmed the sparse/full side is dominated by custom sparse-selection kernels; next I’m extracting the exact SimpleGLA execution path to rank the mixed-layer hotspots across both branches.Optimizing tool selection...

Searched for regex `class SimpleGLAAttnBackend|def forward_extend|def forward_decode|chunk_simple_gla|fused_recurrent_simple_gla|chunk_gated_delta_rule|fused_sigmoid_gating_delta_rule_update|fused_recurrent_gated_delta_rule_update|chunk_kda|fused_recurrent_kda|fused_kda_gate|causal_conv1d_fn|causal_conv1d_update` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 112 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#220-220), lines 220 to 520

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#320-320), lines 320 to 760

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1652

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/fused_recurrent.py#1-1), lines 1 to 260

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/fla/chunk.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py#1-1), lines 1 to 260

I have the lightning branch now. Its hot path is much thinner than the sparse/full branch: state load/store plus one SimpleGLA kernel family, with no conv or top-k stages in MiniCPM-SALA’s actual SimpleGLA mixer. I’m pulling exact call sites next so the final ranking can be tied to concrete files/functions rather than just architecture guesses.

Searched for regex `allocate_and_compress_keys|get_compress_k_v2|get_compress_k_v2_padded|compressed_attention\(|compressed_attention_tilelang|fused_attn_pooling_online_topk_prefill|fused_attn_pooling_online_topk_decode|get_block_table_v2|get_block_table_v3|create_attention_kernel` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 40 results

Searched for regex `class MiniCPMAttention|self.attn\(|RadixAttention\(|MiniCPMLightningMixer|SimpleGLAAttnBackend` (`**/python/sglang/srt/models/minicpm.py`), 24 results

Searched for regex `class SimpleGLAAttnBackend|fused_recurrent_simple_gla|chunk_simple_gla|mode = "fused_recurrent" if seq_len < 64 else "chunk"|initial_state = layer_cache.temporal|layer_cache.temporal\[mamba_indices` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 16 results

Searched for regex `def chunk_gated_delta_rule_fwd|chunk_local_cumsum|chunk_scaled_dot_kkt_fwd|solve_tril|recompute_w_u_fwd|chunk_gated_delta_rule_fwd_h|chunk_fwd_o` (`**/python/sglang/srt/layers/attention/fla/chunk.py`), 28 results

Searched for regex `def fused_recurrent_gated_delta_rule_fwd|def fused_recurrent_gated_delta_rule_fwd_kernel` (`**/python/sglang/srt/layers/attention/fla/fused_recurrent.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#250-250), lines 250 to 380

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#560-560), lines 560 to 1025

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1160-1160), lines 1160 to 1225

Searched for regex `attention_kernel.forward_extend|attention_kernel.forward_decode|self.attention_kernel` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1025-1025), lines 1025 to 1165

Searched for regex `force-dense-minicpm|dense_as_sparse|force_dense_minicpm` (`**/benchmark/soar/demo_sala/**`), 21 results

Searched for regex `force_dense_minicpm|dense_as_sparse` (`**/python/sglang/**`), 38 results

The last gap is not in the source tree: SimpleGLA itself comes from the external `fla` package, so the repo only shows its call boundary, not the internal CUDA/Triton kernels. That is enough to rank priorities: I can now separate “our custom kernels” from “already-external kernels” and recommend where optimization effort is most likely to pay off on 6000D.

Searched for regex `return getattr\(self\.hf_config, "has_sparse_attention", False\) if not self\.force_dense_minicpm else False|return getattr\(self\.hf_config, "sparse_layer_ids", \[\]\) if not self\.force_dense_minicpm else \[\]` (`**/python/sglang/srt/configs/model_config.py`), 4 results

Searched for regex `class MiniCPMHybridConfig|full_attention_layer_ids|sparse_layer_ids|lightning_layer_ids` (`**/python/sglang/srt/configs/minicpm.py`), 16 results

Searched for text `--force-dense-minicpm` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 1 result





Fetched https://soar.openbmb.cn/competition

Fetched https://soar.openbmb.cn/toolkit



Fetched https://soar.openbmb.cn/competition

Completed: *Find mixed attention paths* (1/5)

**Key Finding**

1. Your current demo serving config does not actually run the sparse MiniCPM path. It enables force-dense mode in prepare_env.sh, and the model config then suppresses sparse attention and sparse layer ids in model_config.py and model_config.py.

That matters because there are two different answers:

1. True MiniCPM-SALA hybrid path in source code: 75% lightning + 25% sparse/full.
2. Current submission-like runtime from your demo env: lightning + dense full attention, with sparse kernels largely bypassed.

Below I am answering for the true hybrid path first, then noting what changes if you keep force-dense enabled.

**Operation Review**

1. Lightning layers
   - Layer routing is defined in minicpm.py and selected in minicpm.py.
   - The mixer does QKV projection, optional q/k norm, optional rope, then calls the SimpleGLA backend in minicpm.py.
   - The actual backend is hybrid_linear_attn_backend.py.
   - Runtime sequence:
     1. Load recurrent state from the per-layer temporal cache in hybrid_linear_attn_backend.py.
     2. Choose fused recurrent mode for short sequences or decode, and chunk mode for longer extend/prefill in hybrid_linear_attn_backend.py.
     3. Call fused_recurrent_simple_gla in hybrid_linear_attn_backend.py or chunk_simple_gla in hybrid_linear_attn_backend.py.
     4. Write final recurrent state back in hybrid_linear_attn_backend.py.

2. Sparse/full MiniCPM layers
   - Layer routing starts at minicpm.py and sends attention through RadixAttention in minicpm.py.
   - The model-specific backend is minicpm_backend.py.
   - Runtime sequence in sparse mode:
     1. Save KV.
     2. Build compression metadata and sparse batch split in minicpm_backend.py.
     3. Compress keys with get_compress_k_v2 or allocate_and_compress_keys in minicpm_backend.py and minicpm_backend.py.
     4. Score sparse candidates with compressed_attention or compressed_attention_tilelang in minicpm_backend.py and minicpm_backend.py, optionally through pre-JIT fused topk kernels created in minicpm_backend.py and minicpm_backend.py.
     5. Build sparse page tables with get_block_table_v2 or get_block_table_v3 in minicpm_backend.py and minicpm_backend.py.
     6. Run the final attention kernel through the flashinfer or flashattn abstraction in minicpm_backend.py and minicpm_backend.py.

**Top 3 Targets**

1. Lightning SimpleGLA core kernel family
   - Call sites: hybrid_linear_attn_backend.py and hybrid_linear_attn_backend.py
   - Why it is top 1:
     1. It is used by about 75% of the layers.
     2. It is on both extend and decode paths.
     3. For long prompts, chunk mode is likely the dominant lightning-prefill cost.
     4. For token-by-token decode, fused recurrent mode repeats on every generated token and every lightning layer.
   - Practical note:
     1. The repo only contains the call boundary.
     2. The actual kernel implementation comes from the external fla package, so optimizing this means either vendoring/replacing that kernel path or adding a new in-repo specialized alternative.

2. Sparse candidate scoring kernel family
   - Call sites: minicpm_backend.py, minicpm_backend.py, minicpm_backend.py, minicpm_backend.py
   - Why it is top 2:
     1. It is the extra work that sparse layers pay before they can use the final attention kernel.
     2. It is highly memory-traffic heavy.
     3. It is more likely to have optimization headroom than flashinfer itself.
     4. It matches the article’s core lesson: reduce extra passes and fuse selection logic before the main attention math.

3. Sparse key compression kernel family
   - Call sites: minicpm_backend.py, minicpm_backend.py, minicpm_backend.py, minicpm_backend.py
   - Why it is top 3:
     1. It runs before every sparse topk decision.
     2. It creates large intermediate tensors and extra HBM traffic.
     3. On 6000D-class hardware, reducing global memory movement is usually worth more than squeezing a few percent from an already-tuned final attention kernel.

**What I Would Not Target First**

1. The final flashinfer sparse attention call in minicpm_backend.py and minicpm_backend.py
   - It may still be expensive, but it is already delegated to a mature backend.
   - First-pass ROI is usually better in the custom pre-attention pipeline around it.

2. Sparse block-table construction in minicpm_backend.py and minicpm_backend.py
   - Important, but usually second-order unless profiling shows decode is metadata-bound at small batch.

**6000D-Specific Adaptation Of The Article**

The WeChat article’s methods are directionally useful, but I would adapt them like this for 6000D instead of copying 3080-style tuning:

1. Prioritize launch reduction and HBM-traffic reduction over tiny single-kernel heroics.
   - 6000D has more room for larger working sets and more concurrent CTAs than 3080.
   - The best gain is more likely from fusing stages and reducing intermediate reads and writes.

2. Be careful with very register-heavy kernels.
   - A 3080-oriented optimization can win by using extreme register tiling even if occupancy drops.
   - On 6000D, that can backfire if it reduces parallelism too much across long varlen batches.

3. Use larger-tile and persistent-style variants where batching is stable.
   - Especially for lightning chunk mode and sparse topk over long contexts.
   - This matches the article’s profile to PTX to optimize loop, but the target should be higher sustained throughput across larger batches, not just single-SM micro wins.

4. Fuse metadata transforms into producer kernels where possible.
   - This is the strongest overlap with the article.
   - For MiniCPM sparse layers, compression, pooling, topk, and page-table prep are currently too staged.

5. Treat SimpleGLA as the main architectural opportunity.
   - The toolkit’s technical-path section explicitly notes lightning attention uses an independent linear-attention state path.
   - That means it deserves its own optimization strategy rather than sharing assumptions with full attention.

**Proposal**

Objective and expected gain

1. First optimization feature: specialize the lightning SimpleGLA path for 6000D, with an in-repo fast path that keeps the current backend as fallback.
2. Expected gain:
   1. Best case, this is the highest-impact single feature because lightning layers dominate layer count.
   2. It can improve both long-prefill and decode latency, not just one side.

Rule-compliance check

1. This is allowed by the latest competition page and toolkit:
   1. kernel optimization, KV/memory optimization, and backend optimization are explicitly in scope.
   2. we keep the base MiniCPM-SALA model unchanged.
   3. we do not rely on prefix cache, which is explicitly forbidden in evaluation.
   4. we do not change fixed concurrency evaluation logic.
   5. we keep prepare_env.sh and prepare_model.sh contracts intact.
2. Accuracy/stability guardrail:
   1. correctness coefficient must stay above the 97% threshold.
   2. since this is a runtime-kernel change, expected risk to correctness is low if we preserve numerics and state transitions exactly.

Risk to accuracy and stability

1. Low to medium for a backend-dispatch change with a fallback path.
2. Main risks:
   1. recurrent-state mismatch between old and new lightning kernel
   2. varlen cu_seqlens edge cases
   3. chunk-to-recurrent transition around the 64-token mode switch in hybrid_linear_attn_backend.py

Exact files and functions to change if approved

1. Primary dispatch point:
   - hybrid_linear_attn_backend.py
2. Caller contract to keep stable:
   - minicpm.py
3. Hybrid cache contract to preserve:
   - minicpm.py
4. If that proves too invasive, fallback first feature:
   - optimize the sparse custom pipeline only inside minicpm_backend.py

Validation commands

Correctness:
python3 eval_model_001.py --api_base http://127.0.0.1:30000 --model_path /models/MiniCPM-SALA --data_path <perf_public_set.jsonl> --concurrency 32

Speed:
bash SOAR/bench_serving.sh http://127.0.0.1:30000

Recommended profiling before code changes:
nsys profile --trace=cuda,nvtx -o minicpm_sala_profile <server or benchmark command>
ncu --set full --target-processes all <server or benchmark command>

Success criteria

1. Same model outputs within expected tolerance on public correctness eval.
2. No regression in stability across long-context requests.
3. Measurable duration reduction in at least one of S1, S8, or S∞.
4. No increase in submission-flow complexity beyond prepare_env.sh plus optional prepare_model.sh.

Rollback plan

1. Keep the current backend as the default fallback.
2. Gate the new kernel path behind a server arg or env flag.
3. If correctness or stability regresses, disable the new path without touching model weights or preprocessing outputs.

**Important Decision Before Editing**

Because of prepare_env.sh, I need you to choose which runtime we are optimizing:

1. True hybrid path in source code: lightning plus sparse/full
2. Current submission-like config: lightning plus forced-dense full attention

If you want the highest-value first feature for the current config, I would start with lightning SimpleGLA only.
If you want the highest-value first feature for the source-level hybrid architecture, I would still start with lightning SimpleGLA, and queue sparse topk fusion second.

1. Approve feature 1: 6000D-oriented lightning SimpleGLA fast path
2. Approve feature 1 instead on the sparse topk/compression pipeline
3. Clarify that you want optimization only for the current force-dense runtime, not the true hybrid path
