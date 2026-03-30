I’m turning the prior code review into a proposal-only checklist focused on the current force-dense setup. No edits yet; I’ll keep it to what must be verified before touching the lightning-only fusion path.**Proposal Checklist**

Under the current force-dense setup, the target is only the lightning path in minicpm.py. The sparse minicpm4 path is out of scope for this feature.

**1. Compatibility Items To Verify Before Coding**

1. Confirm the active path is really the lightning mixer for the served model under the current args.
Context:
minicpm.py
model_config.py

2. Confirm the shipped MiniCPM-SALA config still has the required pattern for lightning layers:
qk_norm=true and lightning_use_rope=true, while force-dense is enabled.
Relevant code consumption:
minicpm.py

3. Confirm the current lightning sequence is exactly:
packed qkv projection -> split q/k/v -> q_norm/k_norm -> rope -> reshape -> backend
This is the fusion candidate.
Current sequence:
minicpm.py

4. Confirm the tensor contract matches the existing fused op assumptions:
packed qkv in one contiguous tensor
full-head rotary, not partial rotary
no MRotary-style path
head_dim supported by existing fused kernel
Existing fused contract reference:
qwen3_moe.py
moe.py

5. Confirm MiniCPM lightning uses a head_dim that the existing fused kernel already accepts.
Qwen’s current guard is 64, 128, or 256.
MiniCPM-SALA shipped config uses 128 for lightning, which is favorable.

6. Confirm the qkv activation dtype on the quantized serving path is bf16 at this point in the graph.
The existing fused path is currently guarded by bf16 in Qwen.
Reference:
qwen3_moe.py

7. Confirm RoPE style compatibility.
The existing fused path assumes the same RoPE convention as the target model path:
same base theta
same is_neox handling
same rotary_dim behavior
This must match MiniCPM lightning exactly.

8. Confirm there is no hidden dependency on RadixAttention KV-buffer fusion that the lightning path lacks.
Qwen’s fallback path has fused-set-kv-buffer logic, but the fused qk_norm_rope op itself is not inherently tied to that. We still need to verify there is no implicit coupling in the call pattern.

9. Confirm numerical equivalence requirements before benchmarking.
The fused lightning path must match the current unfused lightning path closely enough that accuracy does not regress beyond evaluation noise.

10. Confirm the optimization surface is still worth it under force-dense.
Because force-dense removes sparse attention, this fusion would hit the currently active lightning-heavy serving path, which is exactly why it is a sensible next feature now.

**2. Insertion Point In minicpm.py**

The exact insertion point is the current lightning pre-backend block in minicpm.py.

That block currently does:

1. qkv projection
2. split q, k, v
3. q_norm and k_norm
4. RoPE on q and k
5. reshape to per-head tensors
6. pass into SimpleGLA backend

The cleanest proposal is:

1. Add a MiniCPMLightningMixer helper similar in spirit to Qwen’s apply_qk_norm_rope path.
2. Call it immediately after qkv projection and before the current explicit q_norm / rope code.
3. Keep the existing unfused path as the fallback.
4. Gate the fused path conservatively, for example by:
bf16 only
head_dim supported
qk_norm enabled
use_rope enabled
no unexpected rope variant mismatch

So the real insertion region is:
minicpm.py

And the design reference is:
qwen3_moe.py

**3. Fallback Criteria: Reuse Existing Fused Op vs Write MiniCPM-Specific Fused Op**

Reuse the existing fused op first if all of the following hold:

1. MiniCPM lightning presents packed qkv in the same basic layout expected by the current fused kernel.
2. head_dim is supported without extra packing hacks.
3. rotary_dim is effectively full-head and matches the kernel’s rotary assumption.
4. q_norm and k_norm are plain RMSNorm with compatible epsilon and weight layout.
5. positions can be passed in the same shape and dtype expected by the fused op.
6. integrating the existing op does not require extra transpose or copy steps that erase the speed gain.
7. numerical output matches the current unfused path closely enough.

Switch from reuse to a MiniCPM-specific fused op if any of these fail:

1. The existing fused op requires a qkv memory layout that forces extra reshapes or copies in the MiniCPM lightning path.
2. RoPE handling differs enough that adapting the existing op becomes messy or fragile.
3. The current fused op is too tied to the Qwen attention call pattern or assumptions around downstream attention integration.
4. MiniCPM lightning needs a different output layout for SimpleGLA, and adapting the existing kernel introduces enough glue code to cancel the benefit.
5. Numerical parity is hard to preserve with the reused kernel.
6. Benchmark profiling shows the reuse path spends too much time on pre/post kernel tensor movement rather than on the fused work itself.

The practical decision rule is:

1. If existing fused op can be inserted with only a small guarded helper in MiniCPMLightningMixer, reuse it.
2. If reuse needs significant tensor marshaling or model-specific hacks, stop and write a MiniCPM-lightning-specific fused op instead.
3. If even a MiniCPM-specific fused op would only touch a minor share of runtime after profiling, abandon this path and return to Marlin or CHANGE_0052 refinement.

**Recommended Approval Scope**

If you approve this direction, the smallest disciplined feature is:

1. Lightning-only fusion
2. Force-dense mode only
3. Reuse existing fused qk_norm_rope first
4. Keep unfused fallback
5. No sparse-path changes in this iteration

That is the narrowest proposal with the best chance of landing as one clean feature rather than turning into a broad model rewrite.
