You’re right to challenge this. I’ll verify directly in the MiniCPM-SALA code paths for `target_verify` and SimpleGLA state handling, then give you a definitive answer based on code evidence.

Searched for regex `target_verify|TARGET_VERIFY|speculative|ngram|draft_token_num` (`**/python/sglang/srt/model_executor/**`), 188 results

Searched for regex `fused_recurrent_simple_gla|hybrid_linear_attn_backend|_store_final_state|query_start_loc` (`**/python/sglang/srt/layers/**`), 200 results

I found the exact hotspot files. Next I’ll read the `hybrid_linear_attn_backend` verify path and confirm whether state rollback is implemented or silently wrong.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#420-420), lines 420 to 455

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1540-1540), lines 1540 to 1703

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#230-230), lines 230 to 290

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#1-1), lines 1 to 260

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#260-260), lines 260 to 450

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#290-290), lines 290 to 380

Searched for regex `retrieve_parent_token|retrieve_next_token|retrieve_next_sibling|track_ssm_h_src|track_ssm_final_src` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 146 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#960-960), lines 960 to 1135

I confirmed something important: the file has two different recurrent paths. One (`gated_delta_rule`) has explicit target-verify handling, but the `SimpleGLA` path still looks unguarded. I’ll now verify which path MiniCPM-SALA actually uses so we can settle this definitively.

Searched for regex `class .*SimpleGLA|class .*Hybrid|fused_recurrent_simple_gla|fused_recurrent_gated_delta_rule_update` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 12 results

Searched for regex `SimpleGLA|lightning|hybrid_linear_attn_backend|minicpm_backend|attention_backend` (`**/python/sglang/srt/models/**`), 103 results

Searched for regex `minicpm.*lightning|SimpleGLA|fused_recurrent_simple_gla` (`**/python/sglang/srt/**`), 60 results

Searched for regex `is_target_verify|target_verify|speculative` (`**/python/sglang/srt/models/minicpm.py`), no results

I’ve confirmed the actual MiniCPM-SALA code path now. I’m updating session memory to correct the earlier over-broad “NGRAM works out-of-the-box” claim so we don’t repeat this mistake.

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/ngram_speculative_decoding_research.md)

You are correct to question it. Based on the current code, I am **not confident NGRAM is safe for MiniCPM-SALA Lightning (SimpleGLA) today**.

What I verified:

1. MiniCPM Lightning layers do use SimpleGLA backend:
minicpm.py
minicpm.py

2. In SimpleGLA metadata for target_verify, query_start_loc is built with step=draft_token_num (non-consecutive stride):
hybrid_linear_attn_backend.py

3. SimpleGLA forward path writes final recurrent state back immediately, with no target_verify rollback logic:
hybrid_linear_attn_backend.py
hybrid_linear_attn_backend.py

4. NGRAM verification does KV rollback/compaction, but I did not find corresponding SimpleGLA temporal-state rollback in that path:
ngram_info.py
ngram_info.py

So the earlier broad statement “NGRAM works out-of-the-box with MiniCPM-SALA” was too strong. For this architecture, your risk description is valid and likely correct.

I also updated session memory to prevent repeating that mistake.

Brief summary: I re-checked the real MiniCPM-SALA code path and confirmed that SimpleGLA target_verify appears to lack safe recurrent-state rollback, so NGRAM should be treated as incompatible/high-risk until fixed and validated.

Continued with Autopilot: Task not yet completeCompleted: I re-verified the MiniCPM-SALA code path and confirmed the risk is real for NGRAM with Lightning/SimpleGLA (target_verify uses strided query_start_loc and SimpleGLA state writeback has no explicit rollback handling), and I corrected session memory to avoid repeating the earlier over-optimistic claim.

Continued with Autopilot: Task not yet completeI finished the verification task end-to-end: validated the MiniCPM-SALA Lightning/SimpleGLA speculative path in code, confirmed NGRAM is high-risk/incompatible without recurrent-state rollback safeguards, and corrected session memory to reflect this so future recommendations won’t repeat the earlier overstatement.


