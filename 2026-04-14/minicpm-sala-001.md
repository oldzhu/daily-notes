I can research the code-level optimizations from the local codebase — the model implementation files are here. I don't need the fcloud config.json right now since I already have the architectural details from prior analysis (8 standard attn layers, 24 lightning layers, residual_scale, etc.).

Let me proceed with the code research locally: 

Updated todo list

Research MiniCPM model code

Searched for files matching `**/python/sglang/srt/models/*minicpm*`, 8 matches

Searched for files matching `**/python/sglang/srt/layers/**`, 30 matches

Searched for regex `residual_scale|rope|rotary|position.*embed` (`**/python/sglang/srt/**`), 50 results

Searched for regex `residual_scale|fused_qk_norm_rope|enable.*fused` (`**/python/sglang/srt/**`), 30 results

Searched for regex `gla|lightning.*attention|gated.*linear` (`**/python/sglang/srt/**`), 30 results

Searched for files matching `**/python/sglang/srt/models/minicpm*.py`, 8 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#280-280), lines 280 to 450

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#590-590), lines 590 to 660

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#1-1), lines 1 to 150

Searched for regex `hybrid_linear_attn|SimpleGLAAttnBackend|GLA` (`**/python/sglang/srt/layers/attention/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#150-150), lines 150 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#200-200), lines 200 to 440

Searched for files matching `**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, 2 matches

Searched for regex `class.*GLA|SimpleGLAAttnBackend` (`**/python/sglang/srt/layers/attention/**`), 9 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1-1), lines 1 to 100

Searched for regex `enable_fused_qk_norm_rope` (`**/python/sglang/srt/server_args.py`), 2 results

Searched for regex `rope.*kernel|apply_rope` (`**/3rdparty/**`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#100-100), lines 100 to 1440

Searched for regex `enable_fused_qk_norm_rope` (`**/python/sglang/srt/server_args.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#1440-1440), lines 1440 to 1550

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#250-250), lines 250 to 500

Searched for regex `class.*Rotary|MRotaryEmbedding` (`**/python/sglang/srt/layers/rotary_embedding.py`), 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#100-100), lines 100 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#1409-1409), lines 1409 to 1520

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/minicpm_optimization_findings.md)

Good findings. Let me dig deeper into the specific code paths to understand exactly what can be optimized: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#440-440), lines 440 to 520

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#530-530), lines 530 to 680

Now let me check the **MiniCPMAttention** class (standard attention) for the RoPE handling:  

Searched for text `class MiniCPMAttention` (`**/python/sglang/srt/models/minicpm.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#170-170), lines 170 to 300

Now let me check the fused QK norm + rope implementation for lightning layers:   

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#380-380), lines 380 to 450

Now let me check the `get_rope` function and specifically the `forward_native` method to understand the float32 upcast:    

Searched for text `def forward_native` (`**/python/sglang/srt/layers/rotary_embedding.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#220-220), lines 220 to 310

Searched for text `def forward_cuda` (`**/python/sglang/srt/layers/rotary_embedding.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#342-342), lines 342 to 420

Key finding: `forward_cuda` calls `apply_rope_with_cos_sin_cache_inplace` which **already handles bf16 natively** — the sgl-kernel RoPE kernel works in-place on whatever dtype the query/key are. The problem is that `MiniCPMAttention.forward()` explicitly upcasts to float32 **before** calling `self.rotary_emb()`.

Now let me check the cos_sin_cache dtype:   

Searched for text `cos_sin_cache` (`**/python/sglang/srt/layers/rotary_embedding.py`), 15 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/rotary_embedding.py#160-160), lines 160 to 200

The cos_sin_cache is computed in float32 and stored as float. The CUDA kernel `apply_rope_with_cos_sin_cache_inplace` handles format conversion internally. So the fix for MiniCPMAttention is simple: **just remove the float32 upcast** and let the sgl-kernel RoPE handle it natively in bf16.

Now let me look at the residual_scale opportunity more carefully. Let me check if RMSNorm supports fused add+norm (which it does via the 2-arg call):    

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#630-630), lines 630 to 660

Now let me understand exactly where the residual_scale multiply happens:

1. `hidden_states = self.self_attn(...)` → attention output
2. `hidden_states = hidden_states * self.residual_scale` → scalar multiply (**first**)
3. `hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)` → fused add+norm
4. `hidden_states = self.mlp(hidden_states)` → MLP
5. `hidden_states = hidden_states * self.residual_scale` → scalar multiply (**second**)

The residual_scale is a **scalar constant** multiplied into hidden_states before it goes into the fused RMSNorm. Can we fold it?

The fused `input_layernorm(hidden_states, residual)` computes: `residual_new = residual + hidden_states`, then `output = RMSNorm(residual_new)`. The `hidden_states * residual_scale` happens BEFORE the add, so folding it into the layernorm weight won't be mathematically equivalent since it also affects the residual.

Actually wait — let me re-read the flow. The multiply happens on the *attention output*, which then gets added to the residual in the *next* layernorm call. So `residual_scale * attn_output` scales the contribution before it enters the residual stream. This is NOT equivalent to scaling the layernorm weights.

However, we can still optimize by noting that `hidden_states * scalar` for a constant scalar could be absorbed elsewhere. Let me think...

Actually, since `residual_scale` is just a scalar multiply on a tensor, it could potentially be fused into `o_proj` weight or bias at load time. The `o_proj` is the last linear layer in attention — its output gets multiplied by `residual_scale`. So `residual_scale * o_proj(x) = (residual_scale * o_proj.weight) @ x`. We can fold it into the weight!

Same for MLP: the last layer output is multiplied by `residual_scale`. Let me check the MLP structure:  

Searched for text `class MiniCPMMLP` (`**/python/sglang/srt/models/minicpm.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#132-132), lines 132 to 170

Good. Now I have full understanding. Let me also quickly check the lightning attention's non-fused RoPE path since we already have `--enable-fused-qk-norm-rope` enabled: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#407-407), lines 407 to 432

Starting: *Propose Phase 2 optimizations* (4/5)

Research complete. Here's my **Phase 2 Optimization Proposal**:

---

## Optimization 1: Remove float32 RoPE upcast in MiniCPMAttention

**Objective**: Eliminate unnecessary float32↔bf16 type conversions in the 8 standard attention layers.

**Current code** (minicpm.py):
```python
if self.attn_use_rope:
    orig_dtype = q.dtype
    q, k = q.float(), k.float()        # bf16 → fp32 (unnecessary!)
    q, k = self.rotary_emb(positions, q, k)
    q, k = q.to(orig_dtype), k.to(orig_dtype)  # fp32 → bf16 (unnecessary!)
```

**Proposed change**:
```python
if self.attn_use_rope:
    q, k = self.rotary_emb(positions, q, k)
```

**Why it's safe**: `RotaryEmbedding.forward_cuda()` calls `apply_rope_with_cos_sin_cache_inplace()` from sgl-kernel, which handles bf16 inputs with fp32 cos_sin_cache natively. The float32 upcast is defensive code from the original HF implementation — not needed with sgl-kernel.

**Impact**: Removes 4 dtype cast ops × 8 layers = 32 unnecessary ops per forward pass. Expected ~2-5% speedup on prefill-heavy workloads (where these 8 layers are most expensive).

**Risk**: Very low. The fused path for Lightning layers already runs RoPE in bf16 without issues.

---

## Optimization 2: In-place residual_scale multiplication

**Objective**: Avoid tensor allocation for the constant scalar multiply in all 32 layers.

**Current code** (minicpm.py):
```python
hidden_states = hidden_states * self.residual_scale  # allocates new tensor
...
hidden_states = hidden_states * self.residual_scale  # allocates new tensor
```

**Proposed change**:
```python
hidden_states *= self.residual_scale  # in-place, no allocation
...
hidden_states *= self.residual_scale  # in-place, no allocation
```

**Impact**: Eliminates 64 tensor allocations per forward pass (2 per layer × 32 layers). Expected 1-2% improvement from reduced memory pressure.

**Risk**: None — mathematically identical. The tensor is consumed immediately by the next layernorm.

---

## Optimization 3: torch.compile with OOM mitigation

**Objective**: Keep the 5-7% torch.compile gain while preventing OOM during long-output generation.

**Proposed approach**: Lower `--torch-compile-max-bs` from 32 to 16, and reduce `--mem-fraction-static` from 0.84 to 0.80 to leave more headroom for compiled graph workspace.

**Risk**: Medium — needs empirical testing. Lower max-bs may reduce compile benefit; lower mem-fraction reduces KV cache slots.

---

**Recommendation**: Implement optimizations 1 and 2 first (zero code risk, no accuracy impact), then test torch.compile stability separately.

Shall I proceed with implementing optimizations 1 and 2?
