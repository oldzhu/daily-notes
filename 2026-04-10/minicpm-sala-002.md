Let me trace the exact scoring flow. 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#497-497), lines 497 to 650

Now let me check how `k` (compressed keys) relate to the full KV cache — specifically the compression step:  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#140-140), lines 140 to 210

Now let me check the compression kernel to understand what it does:   

Searched for text `compress_k_to_scratch_kernel` (`**/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_kernels.py#1-1), lines 1 to 65

Now I have the full picture. Here's how top-k scoring works in this sparse attention:

**It does NOT score against all raw K tokens.** The process is:

1. **Compress K** (`compress_k_to_scratch_kernel`): Average every `kernel_size=32` consecutive K vectors with `stride=16`, producing compressed keys. For a 2000-token sequence, this yields ~124 compressed K1 vectors and ~30 compressed K2 vectors (k2 uses 4× larger kernel/stride).

2. **Score Q vs compressed K** (`infllmv2_attn_stage1`): Compute attention scores `Q @ K_compressed^T / sqrt(head_dim)` — this is a full dot product against all compressed K vectors, not the raw ones. Much cheaper since there are ~124 compressed vectors instead of ~2000 raw tokens.

3. **Max-pool scores into block scores** (`max_pooling_1d_varlen`): Aggregate per-compressed-token scores into per-block scores (block_size=64), taking the max within each block.

4. **TopK selection** (`block_score.topk(topk)`): Pick the top-k blocks from the pooled scores. With `topk=8` and `local_blocks=1`, effective topk=9 → selects 9 blocks × 64 tokens = 576 tokens.

5. **Full attention on selected blocks**: Only compute full Q@K@V attention on the selected 576 tokens (not shown in this function — happens downstream).

**So yes, all compressed scores are computed first, then top-k is picked.** But "all" means ~124 compressed vectors, not ~2000 raw tokens. The compression (averaging 32 K vectors into 1) is where GPTQ errors get partially smoothed but not eliminated — and the binary top-k selection on 0.4% of candidates is where the remaining error becomes fatal.


