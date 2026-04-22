Created 3 todos

Let me check the relevant code sections to document precisely:  

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#860-860), lines 860 to 910

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#80-80), lines 80 to 160

Good, I have all the details. Now let me document this properly and save to memory: 

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/gptq_calibration_attention_mismatch.md)

Now update the optimization catalog to document this finding: 

Replacing 1 lines with 1 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)


```

```


Starting: *Proceed to M1 residual scale folding* (3/3)

Done. The mismatch is documented in:
- **Repo memory**: `/memories/repo/gptq_calibration_attention_mismatch.md`  
- **Optimization catalog**: OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md (G3 section updated with confirmed finding)

Key takeaway: calibration uses **sparse** (HF native), inference uses **dense** (`--force-dense-minicpm`). W4A16 makes the impact small, but if we ever need to squeeze accuracy, requantizing with dense calibration is a known lever.

---

Now for **M1 — Residual Scale Folding**. I need to study the exact math first. Let me read the MiniCPM model code to understand the residual scale pattern: 

Research residual scale folding math

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/layernorm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#100-100), lines 100 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/layernorm.py#100-100), lines 100 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#250-250), lines 250 to 500

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#500-500), lines 500 to 800

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#800-800), lines 800 to 817

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/layernorm.py#100-100), lines 100 to 145

Great research results. Now let me formulate the M1 proposal. The key finding is that GPTQ quantized weights need special handling — we fold into the dequantization scales rather than the raw weights. Let me verify the Marlin scale format: 

Check Marlin GPTQ scale format

Searched for text `GPTQMarlinLinearMethod` (`**/python/sglang/srt/layers/quantization/gptq.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1-1), lines 1 to 100

Searched for text `apply_gptq_marlin_linear` (`**/python/sglang/srt/layers/quantization/gptq.py`), 4 results

Searched for text `load_weights` (`**/python/sglang/srt/models/minicpm.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#562-562), lines 562 to 750

Searched for text `apply_gptq_marlin_linear` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#750-750), lines 750 to 870

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#700-700), lines 700 to 817

Searched for regex `scale_depth|scale_emb|scale_width|dim_model_base|residual_scale` (`**/python/sglang/srt/models/minicpm.py`), 16 results

Searched for regex `embed_tokens|lm_head` (`**/python/sglang/srt/models/minicpm.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#464-464), lines 464 to 550

Searched for regex `modules_to_not_convert|lm_head|embed|inside_layer_modules` (`**/python/sglang/srt/layers/quantization/gptq.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#570-570), lines 570 to 700

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#200-200), lines 200 to 370

Searched for text `marlin_permute_scales` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#720-720), lines 720 to 760

Searched for text `gptq_marlin_gemm` (`**/python/sglang/srt/layers/quantization/marlin_utils.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#311-311), lines 311 to 345

Searched for regex `get_quant_method|is_lm_head|override_quant|prefix.*lm_head` (`**/python/sglang/srt/layers/quantization/gptq.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#480-480), lines 480 to 520

Searched for text `get_linear_quant_method` (`**/python/sglang/srt/layers/quantization/utils.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#360-360), lines 360 to 400

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils.py#269-269), lines 269 to 330

Searched for files matching `**/demo_sala/**/quantize_config*`, no matches

Searched for regex `lm_head.*true|lm_head.*false|"lm_head"` (`**/benchmark/soar/demo_sala/**`), 20 results

Searched for regex `lm_head|embed_tokens|inside_layer_modules|modules_to_not_convert` (`**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`), 1 result

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py#1-1), lines 1 to 50

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/gptq_marlin_scale_format.md)

Now I have everything needed for a complete M1 proposal. Let me also check the exact config values: 

Searched for regex `scale_depth|dim_model_base|scale_emb|scale_width` (`**/python/sglang/srt/models/minicpm.py`), 13 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#585-585), lines 585 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#625-625), lines 625 to 660

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#720-720), lines 720 to 755

Now I have the complete picture. Here's the **M1 Residual Scale Folding Proposal**:

---

## M1 — Residual Scale Folding Proposal

### Objective
Eliminate 66 runtime scalar multiply kernels per forward pass by folding three constant scaling factors into weights at model load time.

### Expected Gain
**1-3%** speedup (primarily from eliminating kernel launch overhead, especially impactful at small batch sizes during decode)

### What Gets Folded

| Scale Factor | Value | Current Code | Fold Into | Kernels Saved |
|-------------|-------|-------------|-----------|---------------|
| `residual_scale` | `scale_depth / √num_hidden_layers` | `h *= self.residual_scale` after attn + after MLP, per layer | `o_proj.scales` and `down_proj.scales` (GPTQ Marlin dequant scales) | 64 (32 layers × 2) |
| `scale_emb` | config constant | `embed * scale_emb` after embedding lookup | `embed_tokens.weight` (full precision, not quantized) | 1 |
| `1/scale_width` | `dim_model_base / hidden_size` | `h / scale_width` before lm_head | `lm_head.weight` (full precision, not quantized) | 1 |

**Total: 66 kernel launches eliminated per forward pass.**

### Mathematical Proof of Correctness

**Residual scale**: The decoder layer computes:
```
h = attn(norm(residual))        # h = o_proj(attention_output)
h *= residual_scale              # scalar multiply
residual_new = h + residual      # fused_add_rmsnorm does this
```
Folding `residual_scale` into `o_proj` scales:
```
h = (residual_scale * o_proj)(attention_output)   # same result, one less kernel
```
For GPTQ Marlin: `output = X @ (scales * Q_int)`, so `scales_new = scales * residual_scale` gives `output_new = residual_scale * output_old`. **Exact, no approximation.**

Same logic applies to `down_proj` (MLP output).

**scale_emb**: `embed_tokens.weight *= scale_emb` → embedding lookup directly returns scaled values.

**scale_width**: `lm_head.weight /= scale_width` → linear projection directly returns divided values. (Or equivalently, `lm_head.weight *= (dim_model_base / hidden_size)`.)

### Risk Assessment
- **Accuracy risk**: **ZERO** — this is mathematically exact scalar multiplication, not an approximation
- **GPTQ safety**: We modify only the `scales` tensor (bf16 float), not the packed integer `qweight`. The Marlin kernel applies `scales` as a multiplicative factor during dequantization — scalar multiplication distributes through this exactly
- **Stability risk**: None — the folded values are identical to what the runtime would compute

### Rule Compliance
- No model architecture changes (forbidden by SOAR)
- No weight structure changes — same tensors, same shapes, just different numeric values
- Purely a load-time optimization — inference produces identical results

### Files to Change

| File | Change |
|------|--------|
| minicpm.py | 1. In `MiniCPMDecoderLayer.__init__`: keep `self.residual_scale` for reference but don't use in forward |
| | 2. In `MiniCPMDecoderLayer.forward()`: remove both `hidden_states *= self.residual_scale` lines |
| | 3. In `MiniCPMModel.__init__` or `MiniCPMForCausalLM.__init__`: fold `scale_emb` into `embed_tokens.weight` |
| | 4. In `MiniCPMForCausalLM.forward()`: remove `* scale_emb` and `/ scale_width` |
| | 5. Add `fold_scaling_factors()` method called after `load_weights()` — multiplies GPTQ scales of `o_proj`/`down_proj` by `residual_scale`, multiplies `embed_tokens.weight` by `scale_emb`, divides `lm_head.weight` by `scale_width` |

### Implementation Detail: GPTQ Scale Modification

The folding must happen **after** weights are loaded but it can be either before or after `marlin_permute_scales` since scalar multiplication commutes with permutation: `α × permute(S) == permute(α × S)`.

Best hook point: override or extend `load_weights()` to call `_fold_scaling_factors()` at the end.

```python
def _fold_scaling_factors(self):
    residual_scale = self.model.layers[0].residual_scale  # same for all layers
    
    for layer in self.model.layers:
        # Fold into o_proj and down_proj GPTQ scales
        for proj_name in ["self_attn.o_proj", "mlp.down_proj"]:
            proj = layer.get_submodule(proj_name)
            if hasattr(proj, "weight_scale"):  # Marlin format
                proj.weight_scale.data *= residual_scale
            elif hasattr(proj, "scales"):
                proj.scales.data *= residual_scale
    
    # Fold scale_emb into embedding
    self.model.embed_tokens.weight.data *= self.config.scale_emb
    
    # Fold scale_width into lm_head
    if not self.config.tie_word_embeddings:
        self.lm_head.weight.data /= self.scale_width
    # If tied, both scale_emb and 1/scale_width apply to the same weight
```

### Caveat: `tie_word_embeddings`
If `embed_tokens` and `lm_head` share the same weight tensor (`tie_word_embeddings=True`), we can't fold `scale_emb` and `1/scale_width` independently since they'd both modify the same tensor. Need to check the MiniCPM-SALA config for this. **We'd need the config.json from fcloud to verify.**

### Validation
1. **Correctness**: Run accuracy eval — must get identical results (within floating-point noise)
2. **Speed**: Run S1/S8/Smax benchmarks — expect 1-3% improvement
3. **Numerical check**: Compare logits of first 10 tokens between folded and unfolded versions

### Test Commands
```bash
python3 scripts/fcloud/fcloud_workflow.py full  # sync + restart + accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### Success/Failure Criteria
- **Success**: Accuracy unchanged (within ±0.5%), any measurable speed improvement
- **Failure**: Accuracy drops >1% (would indicate a bug in folding logic)

---

Before implementing, I need to verify one thing from the config: **`tie_word_embeddings`**. Do you want to start fcloud so I can check, or do you happen to know whether MiniCPM-SALA ties embeddings?

