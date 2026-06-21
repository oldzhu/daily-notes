Good questions. Let me address them and fix the report:

Searched for text `force_dense` (`**/python/sglang/srt/models/minicpm.py`), no results

Searched for text `force_dense_minicpm` (`**/python/sglang/srt/**/*.py`), 20 results

Let me clarify all three questions:

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/model_config.py#233-233), lines 233 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#658-658), lines 658 to 720

Now I have the full picture. Here are the answers:

## Question 2: Why QKV and O are "sparse only"

**`--force-dense-minicpm` does NOT convert lightning layers to standard attention.** It only affects `model_config.has_sparse_attention` (returns `False`) and `sparse_layer_ids` (returns `[]`). The module class is determined by `config.mixer_types[layer_id]` at init time, which is completely independent of `--force-dense-minicpm`.

So even with `--force-dense-minicpm`:
- **8 `minicpm4` layers** → `MiniCPMAttention` → QKV/O tagged `_soar_w4a8_eligible=True` → **W4A8 YES**
- **24 `lightning` layers** → `MiniCPMLightningMixer` → QKV/O NOT tagged → **W4A8 NO**

The report was accurate — I need to add this clarification though.

## Question 3: QKV/O difference between sparse vs lightning

| | Sparse Attn (8 layers) | Lightning Attn (24 layers) |
|---|---|---|
| **Module class** | `MiniCPMAttention` | `MiniCPMLightningMixer` |
| **Q heads** | 32 × head_dim=128 = **4096** | 16 × head_dim=**64** = **1024** |
| **KV heads** | 8 × 128 = **1024** | 16 × 64 = **1024** |
| **QKV proj shape** | 4096 → **6144** | 4096 → **3072** |
| **O proj shape** | **4096** → 4096 | **1024** → 4096 |
| **W4A8 eligible?** | ✅ Yes | ❌ No |
| **Attention kernel** | FA3 (dense with `--force-dense`) | FLA chunk/recurrent GLA |
| **KV Cache?** | ✅ Paged KV (FP8 e5m2) | ❌ Recurrent state |

The lightning layers have fundamentally different architecture (GLA linear attention with recurrent state), not just "dense instead of sparse."

Now let me fix the report — create a proper Chinese translation and add the `--force-dense-minicpm` clarification:

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_inference_flow_analysis.en.md#1-1), lines 1 to 30




```

```
Now let me create the proper Chinese version:

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_inference_flow_analysis.zh.md)

Ran terminal command:  rm /home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_inference_flow_analysis.zh.md

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_inference_flow_analysis.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add -A && git commit -m "docs: update EN analysis + add proper ZH translation

Changes:
- EN: add --force-dense-minicpm clarification (does NOT convert lightning layers)
- EN: add table comparing sparse vs lightning QKV/O dimensions/eligibility
- ZH: proper Chinese translation with all key findings
- Clarify that MLP W4A8 covers all 32 layers, QKV/O only 8 sparse layers" && git push minicpm-src mixed_minicpm_cudagraph 2>&1

Done. Here's a summary of the answers to your three questions:

## Answers

### 1. No Chinese version → Fixed ✅

The initial `.zh.md` was just a raw copy of the EN file. Now replaced with a proper Chinese translation at RESEARCH_inference_flow_analysis.zh.md.

### 2. Why QKV/O "sparse only"? 

**`--force-dense-minicpm` does NOT convert lightning layers to standard attention.** It only sets `model_config.has_sparse_attention=False` (see `model_config.py:238`). The module class is chosen by `config.mixer_types[layer_id]` at init time:

```
mixer_type == "minicpm4" → MiniCPMAttention      (8 layers, W4A8 ✅)
mixer_type == "lightning" → MiniCPMLightningMixer (24 layers, W4A8 ❌)
```

`MiniCPMLightningMixer` explicitly does **NOT** set `_soar_w4a8_eligible` on its QKV/O linears (unlike `MiniCPMAttention` which does). This is by design — the lightning layers have different head dimensions (16×64 vs 32×128), use GLA recurrent attention, and were excluded from W4A8 scope.

**However**: MLP linears (`gate_up_proj`, `down_proj`) use the same `MiniCPMMLP` class for ALL 32 layers, so W4A8 covers 100% of MLP projections.

### 3. Sparse vs Lightning — QKV/O differences

| | Sparse Attn (8 layers) | Lightning Attn (24 layers) |
|---|---|---|
| **Class** | `MiniCPMAttention` | `MiniCPMLightningMixer` |
| **Q dim** | 32 × **128** = 4096 | 16 × **64** = 1024 |
| **KV dim** | 8 × 128 = 1024 | 16 × 64 = 1024 |
| **QKV shape** | 4096→**6144** | 4096→**3072** |
| **O shape** | **4096**→4096 | **1024**→4096 |
| **Attention** | FA3 dense | GLA recurrent |
| **KV Cache** | ✅ FP8 paged | ❌ Recurrent state |
| **W4A8?** | ✅ | ❌ |

The lightning layers are fundamentally different — not just "dense instead of sparse." They use GLA linear attention with recurrent state, different head dimensions, and no KV cache.

