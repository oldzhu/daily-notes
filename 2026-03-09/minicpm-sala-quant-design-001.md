Excellent evidence. Based on your dump + paper + earlier grep, we can design this cleanly.

Estimated layer groups (32 layers)
From your observed o_gate presence, sparse InfLLM-V2 layers are likely:

0, 9, 16, 17, 22, 29, 30, 31 (8/32 = 25%)
The other 24 are Lightning Attention layers.

Quant module list (recommended)
Include in all layers
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
Reason:

These are major Linear weights and dominate speed/memory benefit.
Present and quantizable across both attention families (shape may differ, still linear).
Include in sparse-only layers
self_attn.o_gate
Reason:

Exists only in InfLLM-V2 sparse blocks; should be quantized there for full sparse-path gain.
Include in linear-only layers
self_attn.z_proj (if consistently present in Lightning blocks)
Reason:

Lightning-specific projection; contributes to linear-attn path compute.
Exclude (for now)
self_attn.rotary_emb
self_attn.compress_k, self_attn.compress_k2
norms: q_norm, k_norm, o_norm, input_layernorm, post_attention_layernorm
embeddings/lm_head (for first quant feature)
Reason:

Not standard linear GEMM targets or too accuracy-sensitive for initial stable quant feature.
Layer-aware GPTQ feature design (what we should build next)
Goal
Avoid mixed-architecture module-mismatch crashes while preserving quant coverage on valid modules.

Design
Discovery pass in preprocess_model.py
Load model in meta/init mode (same trust/attn settings).
For each model.model.layers[i], inspect self_attn submodule names.
Build:
sparse_layers (has o_gate)
linear_layers (has z_proj and no o_gate)
Generate dynamic rules
Keep common module set quantized for all layers.
Add per-layer skip rules only where module is absent:
skip self_attn.o_gate on non-sparse layers
skip self_attn.z_proj on non-linear layers
Optional: log generated rules to JSON in output dir for reproducibility.
Quantization execution
Run GPTQ with generated dynamic config/rules.
Keep current retry-on-failure fallback for unexpected missing module names.
Safety controls
Env toggles:
SOAR_GPTQ_LAYER_AWARE=1
SOAR_GPTQ_ENABLE_Z_PROJ=1
SOAR_GPTQ_ENABLE_O_GATE=1
Why this is better than global skip o_gate
Global skip is safe but loses quantization on valid sparse layers.
Layer-aware rules preserve more speed potential without mismatch crashes.
If you approve, I can implement this as one feature iteration (CHANGE_0030) in preprocess_model.py + EN/ZH docs.


