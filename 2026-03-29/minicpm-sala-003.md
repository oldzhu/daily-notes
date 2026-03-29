I’m reviewing the MiniCPM hybrid routing and attention implementation directly to verify the “sparse removes RoPE” claim, and then I’ll map what your two server flags actually force in this repo so the answer is concrete.

Searched for regex `dense-as-sparse|force-dense-minicpm|force_dense|dense_as_sparse|lightning_use_rope|attn_use_rope|qk_norm|mixer_type|InfLLM|sparse attention|remove RoPE|rope` (`**/python/sglang/srt/**`), 200 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 260

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#260-260), lines 260 to 520

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#560-560), lines 560 to 660

I’ve confirmed the core MiniCPM attention flow; now I’m locating the flag handling so I can say exactly how `--dense-as-sparse` and `--force-dense-minicpm` affect the hybrid routing and whether they block the fusion path.

Searched for regex `dense_as_sparse|force_dense_minicpm|force-dense-minicpm|dense-as-sparse` (`**/python/sglang/srt/**`), 45 results

Searched for regex `attn_use_rope|lightning_use_rope|qk_norm|mixer_types|sparse` (`**/python/sglang/srt/models/minicpm.py`), 64 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#200-200), lines 200 to 320

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1-1), lines 1 to 320

I found the flag entry points. One more code pass will let me separate what is model routing versus what is backend batching behavior, which is the key distinction for whether fusion is still applicable.

Searched for regex `attn_use_rope|lightning_use_rope|qk_norm|has_sparse_attention|sparse_layer_ids|mixer_types` (`**/config*.json`), no results

Searched for regex `attn_use_rope|lightning_use_rope|qk_norm|mixer_types|sparse_layer_ids|has_sparse_attention` (`**/benchmark/soar/demo_sala/**`), 100 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/model_config.py#220-220), lines 220 to 260

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#210-210), lines 210 to 410

I have the main answer now: the flags do not automatically kill the fusion path, but they do change which attention family is even active. I’m doing one final read on the hybrid config class so I can be precise about whether “sparse layers remove RoPE” is encoded as a model-config fact here or only as an external claim.

Searched for regex `MiniCPMHybridConfig|mixer_types|lightning_use_rope|attn_use_rope|has_sparse_attention|sparse_layer_ids` (`**/python/sglang/srt/**`), 116 results

Searched for regex `lightning_use_rope|attn_use_rope` (`**/python/**`), 18 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#1470-1470), lines 1470 to 1525

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/minicpm.py)

The repo code already shows that RoPE is configurable per attention family, not hard-coded away in sparse layers. I’m checking the upstream model config next; that will tell us whether MiniCPM-SALA’s shipped config actually disables RoPE for sparse layers in practice, which is the missing piece.





Fetched https://huggingface.co/openbmb/MiniCPM-SALA/raw/main/config.json

Fetched https://huggingface.co/openbmb/MiniCPM-SALA/blob/main/config.json

The upstream config answered the missing part: in the shipped MiniCPM-SALA config, sparse layers do disable RoPE while lightning layers keep it on. I’m pulling the existing Qwen fused path now so I can contrast its assumptions with MiniCPM-SALA and answer whether a custom fused op is realistic.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/qwen3_moe.py#470-470), lines 470 to 620

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/moe.py#240-240), lines 240 to 290

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1438-1438), lines 1438 to 1492

I now have the exact shipped MiniCPM-SALA config and the existing fused-kernel contract. I’m pulling the final flag semantics from the server-args handling so I can answer which parts are model topology versus backend behavior, and whether a custom MiniCPM-SALA fused op is realistic if reuse fails.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#990-990), lines 990 to 1026

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1508-1508), lines 1508 to 1534

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#4010-4010), lines 4010 to 4032

Yes, for the current shipped MiniCPM-SALA config, that statement is effectively true.

The repo code plus the upstream model config confirm three separate facts. First, MiniCPM-SALA is really hybrid: the model uses per-layer `mixer_types`, and the config class explicitly distinguishes `minicpm4` layers from `lightning-attn` layers in minicpm.py. Second, the shipped Hugging Face config for `openbmb/MiniCPM-SALA` sets `attn_use_rope=false`, `lightning_use_rope=true`, and `qk_norm=true`, with a `mixer_types` list that is 24 lightning layers and 8 `minicpm4` layers, so the 75% / 25% split is consistent with the actual model config. Third, in this repo’s implementation, the `minicpm4` path and the lightning path are not symmetric: the `minicpm4` attention class in minicpm.py does `qkv -> optional RoPE -> attention`, but it does not do `q_norm/k_norm` at all; the lightning mixer in minicpm.py does exactly `qkv -> q_norm/k_norm -> RoPE -> backend`. So for the current model, `qk_norm_rope` is a real match only for the lightning layers. For the `minicpm4` layers, it is not just “RoPE disabled by config”; the class also lacks the q/k norm step entirely.

On your flags: `--force-dense-minicpm` does not make `qk_norm_rope` fail on the lightning layers. What it does is disable MiniCPM’s sparse-attention mode at the model-config level and switch the backend away from the MiniCPM-specific sparse backend. You can see that in model_config.py, where `has_sparse_attention` and `sparse_layer_ids` become false/empty when `force_dense_minicpm` is enabled, and in server_args.py, where `minicpm_flashinfer` is rewritten to plain `flashinfer` under that flag. So `--force-dense-minicpm` mainly means “do not use the sparse MiniCPM backend”. It does not change the lightning-layer math pattern, so it does not block fusion there. `--dense-as-sparse` is different: it only affects the MiniCPM sparse backend’s batch classification logic, making even short batches go through sparse handling, as shown in minicpm_backend.py and minicpm_sparse_utils.py. In your current runtime, where `--force-dense-minicpm` is also set, `--dense-as-sparse` is effectively close to inert because the sparse backend path is being disabled upstream.

On the existing SRT fused op: the current fused kernel used by Qwen is exposed in moe.py, and Qwen’s use site is in qwen3_moe.py. That path is less model-specific than it first sounds: it mutates a packed `qkv` tensor in place and takes generic arguments like number of q/k/v heads, head dimension, norm weights, rope base, and positions. For MiniCPM-SALA lightning layers, the structural match is actually promising: packed `qkv`, `qk_norm=true`, `lightning_use_rope=true`, and head dimension `128` all line up well. So the first attempt should not be “write a brand-new fused kernel”. The first attempt should be to adapt the existing fused `qk_norm_rope` fast path into minicpm.py for the lightning mixer only. If that does not fit cleanly, then yes, implementing our own MiniCPM-SALA-specific fused op is technically feasible. The natural scope would be a specialized lightning-layer kernel that fuses `split qkv + q_norm + k_norm + RoPE`, and possibly reshaping into the backend-friendly layout. That is a valid route, but it is a bigger iteration than reusing the existing kernel entry point.

So the practical conclusion is:

1. For current MiniCPM-SALA, `qk_norm_rope` optimization is only a candidate on the lightning 75% path.
2. Your current `--force-dense-minicpm` flag does not block that candidate; it mostly removes the sparse backend path.
3. `--dense-as-sparse` is not the thing that decides fusion applicability here, and under `--force-dense-minicpm` it is likely giving you little or no effect.
4. If the existing fused SRT kernel cannot be adapted cleanly, a custom MiniCPM-SALA fused op is absolutely possible.

If you want, the next step can be a proposal-only checklist for the lightning-only fusion path:
1. exact compatibility items to verify before coding
2. the insertion point in minicpm.py
3. the fallback criteria for switching from “reuse existing fused op” to “write MiniCPM-specific fused op”
