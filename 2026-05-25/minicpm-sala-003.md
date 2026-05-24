Searched for regex `NVFP4_DEFAULT_CFG|nvfp4|NVFP4` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 20 results

Searched for regex `NVFP4_DEFAULT_CFG|mtq\.|quant_cfg|input_quantizer|disable.*input|enable.*=.*False` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 18 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1530-1530), lines 1530 to 1575

Short answer: because we ship the **stock `mtq.NVFP4_DEFAULT_CFG`** unchanged, and that config quantizes **both** the weight and the activation tensor on every `nn.Linear`. The champion's "NVFP4 + Marlin W4A16 decode" path keeps activations in BF16/FP16 and only stores weights in FP4 E2M1.

Evidence in our code — preprocess_model.py:

```python
config = _copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
quant_cfg = config.setdefault("quant_cfg", {})
for pattern in _nvfp4_default_exclude_patterns():
    quant_cfg[pattern] = {"enable": False}      # only excludes lm_head + gating projections
```

`NVFP4_DEFAULT_CFG` in modelopt 0.43.0 declares (paraphrased):
- `*weight_quantizer*` → `num_bits=(2,1), block_sizes={-1:16}, axis=None` → FP4 E2M1, block 16
- `*input_quantizer*`  → `num_bits=(2,1), block_sizes={-1:16}, axis=None` → **also FP4 E2M1, dynamic per‑forward**
- `*output_quantizer*` → disabled

Because the input_quantizer is left enabled, modelopt:
1. inserts an `input_quantizer` module before every Linear at calibration,
2. records the per‑tensor `input_scale`,
3. writes a checkpoint whose `hf_quant_config.json` advertises `quant_algo="NVFP4"` (which sglang's `ModelOptFp4Config` interprets as **W4A4**),
4. and stores `input_scale` / `alpha = input_scale × weight_scale_2` per Linear.

At runtime sglang's `ModelOptFp4LinearMethod.apply` therefore **must** quantize activations on every forward — that's the `fp4_quantize(x, layer.input_scale_inv)` we saw in the iter‑7 deep‑dive. There is no W4A16 fallback in the modelopt_fp4 loader; the kernel signature itself takes uint8 A.

The champion's recipe ("GPTQ + NVFP4 + Marlin W4A16 decode") avoids this by treating NVFP4 as a **weight‑only storage format** and dispatching through a Marlin‑compatible W4A16 kernel. To match it from our side there are two routes:

1. **Easy (no kernel work)** — flip our config to weight‑only NVFP4 so the exported checkpoint is W4‑weight + BF16‑activation, then load it through a W4A16 path. In modelopt that's done by either:
   - using `mtq.NVFP4_KV_REAL_QUANT_CFG`‑style **weight‑only variants**, or
   - editing the deep‑copied config before `mtq.quantize`:
     ```python
     for k in list(quant_cfg):
         if "input_quantizer" in k:
             quant_cfg[k] = {"enable": False}
     ```
   The catch: sglang's `ModelOptFp4LinearMethod.apply()` doesn't currently have a W4A16 branch — it asserts `x_fp4.dtype == torch.uint8`. So we'd also need a loader/apply path that consumes FP4 weights with BF16 activations. That's the "Marlin W4A16 decode" piece — i.e. a kernel that dequantizes FP4 → BF16 inline like Marlin does for INT4.

2. **What the champion did** — repack the FP4 E2M1 weights into Marlin's INT4‑style packed layout and reuse `gptq_marlin_gemm` (which is W4A16 by construction). FP4 E2M1 ↔ INT4 is a fixed 16‑level remap; the per‑group scale layer takes the FP8 block scale × per‑tensor scale folded into a per‑group FP16 scale. That gives them the FP4 calibration accuracy AND the Marlin small‑M tile picker for decode — exactly the two things our deep‑dive flagged as the weak spots of our current NVFP4 path.

So the W4A4 in our pipeline is not a deliberate design choice — it's a default we never overrode. If we ever revisit NVFP4, **the very first change should be disabling the input_quantizer** in `NVFP4_DEFAULT_CFG`, plus building (or borrowing) a W4A16 FP4‑weight loader / Marlin‑repack path. That single change should recover most of factor §3.1 in CHANGE_0151_phase_b_four_over_six_007 (the W4A4‑vs‑W4A16 accuracy term), and the Marlin‑repack path also recovers most of the §4.2 small‑M speed term. No code change in this turn — just answering the question.
