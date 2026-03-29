I do not agree on that and I already done tuning  SGLang runtime scheduler settings before and it does not give the bechmark improving. I would like to continue the improving at the below 3 :
1.  02 算子融合 - mentioned by 4th week champion post. 
队友擅长英伟达 GPU 上的算子优化，所以在量化模型没出来之前，我们先对模型的前向计算做了一遍梳理，识别到一些可以融合、调优的算子，也对一些算子做了重写。比如原始代码中 RMSNorm 归一化 + RoPE 旋转位置编码，是典型的可以融合的算子，sglang 库也提供了相应的融合算子库可以适配利用，将这类算子融合后可以拿到一定的收益。
2. continue to imprving sgl-kernel optimization - as I did see the speed improving in my fcloud instance but in the official site we did see speed worse - that could be caused by the different benchmark testing data set used in local fcloud and the offcial site - but at least that prove  the 6000D / SM120 GPU 做 Marlin 的 tile / warp 调优(change 0043)  has impact on the speed.
3. Continue the testing and improving the change 0052 - force-dense-lightning-runtime-optimization
pls let me know what do u think 
========
I’m checking the existing fusion and kernel paths before answering so the recommendation is grounded in this repo rather than generic advice. Then I’ll rank your three directions by likely value and implementation risk under the current accuracy-stable setup.

Searched for files matching `**/docs/soar_2026_changes/CHANGE_0043*`, 2 matches

Searched for regex `RMSNorm|RoPE|rope|rmsnorm|fused.*rope|fused.*norm|marlin|tile|warp|LIGHTNING_RECURRENT_THRESHOLD|FAST_STATE_IO` (`**/python/**`), 200 results

Searched for regex `RMSNorm|RoPE|rope|rmsnorm|fused.*rope|fused.*norm|marlin|tile|warp|LIGHTNING_RECURRENT_THRESHOLD|FAST_STATE_IO` (`**/sgl-kernel/**`), 200 results

Search complete: Explore hot paths

Searched for regex `MiniCPM-SALA|RMSNorm|RoPE|Marlin|force-dense|lightning|CHANGE_0043|CHANGE_0052`, 200 results

Searched for files matching `**/*marlin*tuning*`, no matches

Searched for regex `CHANGE_0043|marlin.*tune|marlin.*tuning|marlin.*tile|marlin.*warp`, 96 results

I’ve got enough context to respond, but I want one more read on whether the existing fused norm+rope kernel is already wired into the SRT serving path or only into other runtimes. That decides whether operator fusion is a near-term integration or a deeper porting task.

Searched for regex `fused_qk_norm_rope|apply_rope_with_cos_sin_cache_inplace|fused_add_rmsnorm|rmsnorm\(` (`**/python/sglang/srt/**`), 200 results

Searched for regex `MiniCPM|minicpm_sala|RMSNorm|rotary|rope|q_norm|k_norm` (`**/python/sglang/srt/models/**`), 200 results

Searched for text `fused_qk_norm_rope` (`**/python/**`), 27 results

I confirmed the important constraint: SGLang already has a fused `qk_norm + rope` path in the SRT stack, but it’s wired for other models, not MiniCPM-SALA. That makes operator fusion a real direction, but it is not a trivial “flip a flag” for this model family.I agree with your pushback on scheduler tuning. Given your prior experiments, I would not spend the next iteration there.

My view on your three directions is:

1. Operator fusion is the highest-upside direction, but only if we target the actual MiniCPM-SALA SRT hot path rather than copying the champion blog example too literally.
2. Continuing the sgl-kernel Marlin work is still justified, because you already observed local gains and the official regression can plausibly be workload-shape sensitivity rather than “no real effect”.
3. Continuing `CHANGE_0052` is still worth doing, but as a secondary branch unless we can identify a tighter hotspot than the current recurrent-threshold and state-IO tweaks.

The key nuance is that the repo does already contain fused norm/rope infrastructure in the SRT path, but not obviously for MiniCPM-SALA. There is an existing fused `qk_norm_rope` kernel exposed in moe.py and a server arg switch in server_args.py, and it is actively used by another model in qwen3_moe.py. That matters because it means “operator fusion” is not speculative in this codebase. But the champion’s RMSNorm+RoPE example is not yet a direct MiniCPM-SALA drop-in from what I can see. So if we choose fusion next, I would frame it as: port an already-supported fused `q_norm + k_norm + rope` style optimization into the MiniCPM-SALA attention path if the model structure is compatible, rather than starting with a brand-new fused RMSNorm+RoPE kernel.

On the Marlin side, your argument is technically sound. `CHANGE_0043` already established a real SM120 tuning surface in gptq_marlin.cu and the SRT side carries Marlin tile metadata through parameter.py and marlin_utils.py. If fcloud improved and official worsened, that does not invalidate the optimization. It suggests the candidate table may be overfit toward one matrix-shape mix. That is exactly the kind of issue that can happen when public and private benchmark distributions differ. So I would not abandon `CHANGE_0043`; I would refine it by targeting shape robustness rather than only peak local win rate.

For `CHANGE_0052`, I think it remains valid but is probably the third choice right now. The reason is not that it is wrong. It is that you already have one iteration there, and the next useful step needs better hotspot evidence than “keep tuning thresholds”. The existing optimization in hybrid_linear_attn_backend.py and the env wiring in prepare_env.sh are reasonable, but I would only continue there if we can identify a specific remaining bottleneck inside the force-dense lightning path, not just continue parameter poking.

So my ranking would be:

1. MiniCPM-SALA operator fusion in the SRT path
2. Marlin sgl-kernel robustness tuning for SM120
3. Further `CHANGE_0052` lightning optimization

The main reason fusion goes first is that it can improve the unfused forward path repeatedly across requests and is more likely to transfer across benchmark datasets than a narrowly tuned scheduler or candidate table. The main reason Marlin goes second is that it is still promising, but needs more shape-aware tuning discipline. The reason `CHANGE_0052` goes third is that it likely needs profiling evidence before another clean iteration.

If you want the strongest next single-feature proposal, I would propose this one next, without editing yet:

Objective: adapt existing SRT fused `qk_norm_rope` capability to MiniCPM-SALA attention if its q/k projection and rope layout are compatible.

Expected gain: moderate speed upside on pre-attention hot path with low correctness risk if implemented as a guarded fast path.

Risk:
- medium implementation risk
- low mathematical risk if the fused path preserves exact semantics
- main blocker is MiniCPM-SALA compatibility, not kernel availability

Files I would inspect/change for that proposal:
- minicpm.py
- likely MiniCPM-SALA model definition file in the same SRT model area once confirmed
- rotary_embedding.py
- layernorm.py
- reference implementation in qwen3_moe.py

If you prefer, I can now do the required proposal-only deep dive for one of these three tracks and give you a concrete approved-change plan before any code edits. My recommendation is to start with `1`, and keep `2` as the fallback if MiniCPM-SALA turns out not to match the existing fused kernel interface cleanly.
