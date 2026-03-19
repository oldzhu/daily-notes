I’m checking how the GPTQ module targeting is applied in your preprocessing code and whether `self_attn.o_gate` / `self_attn.z_proj` are structurally safe to quantize for MiniCPM-SALA, so I can answer in terms of both runtime correctness and implementation compatibility.

Searched for regex `SOAR_GPTQ_INCLUDE_MODULES|SOAR_GPTQ_EXCLUDE_MODULES|include_modules|exclude_modules` (`**/benchmark/soar/demo_sala/**`), 63 results

Searched for regex `o_gate|z_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj` (`**/python/sglang/srt/models/minicpm.py`), 72 results

Searched for regex `o_gate|z_proj|include_modules|exclude_modules` (`**/python/**`), 120 results

I found the MiniCPM module definitions and the preprocessing logic. I’m opening the exact rule builder now so I can answer whether moving `o_gate` and `z_proj` into the include list is merely a quality tradeoff or whether it changes how the quantizer targets layers.

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#160-160), lines 160 to 270

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#150-150), lines 150 to 205

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#280-280), lines 280 to 380

I found an important detail in your preprocessing code: the current layer-aware rule builder only turns `exclude_modules` into dynamic skip rules. I’m checking the custom GPTQ MiniCPM adapter next, because whether `o_gate` and `z_proj` get quantized is likely controlled there more than by the include list alone.

Searched for files matching `**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`, 1 match

Searched for regex `simple_layer_modules|inside_layer_modules|modules_in_block_to_quantize|o_gate|z_proj` (`**/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py)

Not safely as a simple env-var change, and probably not in the way you expect.

The key point is: in your current MiniCPM-SALA GPTQ flow, `self_attn.o_gate` and `self_attn.z_proj` are intentionally modeled as **native / non-quantized helper modules** in the custom GPTQ adapter, not just excluded by the shell env vars.

The strongest evidence is in gptqmodel_minicpm_sala.py:

```python
# We keep optional gating/norm helper modules in the tree with :! so GPTQ's
# traversal understands the structure, but they remain native weights.
```

And the module tree explicitly marks them this way:
- gptqmodel_minicpm_sala.py
- gptqmodel_minicpm_sala.py

That means:
- `z_proj:1:!`
- `o_gate:1:!`

So even before talking about your env vars, the custom GPTQ model definition is already saying:
- keep these modules visible to traversal
- but do **not** quantize them

There is a second important detail in preprocess_model.py:
- `_build_dynamic_rules()` ignores `include_modules`
- it only turns `exclude_modules` into skip rules

So today:
- `SOAR_GPTQ_EXCLUDE_MODULES` does matter as an extra guard
- but `SOAR_GPTQ_INCLUDE_MODULES` does **not** by itself force those modules to become quantized targets in the layer-aware path

So if you simply:
- remove `SOAR_GPTQ_EXCLUDE_MODULES`
- add `self_attn.o_gate,self_attn.z_proj` into `SOAR_GPTQ_INCLUDE_MODULES`

the likely outcome is:
- you will **not** reliably quantize those modules anyway
- because the MiniCPM-SALA GPTQ adapter still marks them as native via `:!`

So the short answer is:

1. **Can you do it just by env vars?**
- probably no, not effectively

2. **Would it cause an error immediately?**
- probably not just from changing the env vars
- but it also likely would not achieve the intended quantization

3. **If you modify the adapter to really quantize them, could that cause problems?**
- yes, absolutely possible

Why those two modules are special:
- minicpm.py shows `o_gate` is used as a sigmoid gating term on attention output
- minicpm.py shows `z_proj` is also used as a sigmoid gating term in the lightning mixer path

So they are not ordinary large projection layers like `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
They directly modulate outputs through:
- `attn_output = attn_output * F.sigmoid(o_gate_output)`
- `o = o * F.sigmoid(z)`

That makes them more accuracy-sensitive than their parameter count suggests.

So if you really force-quantize them, the risks are:
- larger accuracy drop than expected
- possible instability in the hybrid attention / lightning path
- and only a small model-size win, because these helper projections are not the main parameter budget

So from an ROI perspective, quantizing them is probably a bad trade:
- small size reduction
- potentially outsized accuracy loss

Would it cause any error in gptqmodel_minicpm_sala.py?
- If you only change env vars: likely no direct error, but also likely no real effect
- If you edit gptqmodel_minicpm_sala.py to remove the `:!` markers and truly quantize them: it could still run, but that would be a new quantization feature and should be treated as a higher-risk experiment, not a harmless extension

My recommendation:
- do **not** target `self_attn.o_gate` and `self_attn.z_proj` next
- if you want to improve 4-bit compression further, first focus on recovering accuracy in the existing quantized set
- those two helper gating modules were almost certainly excluded on purpose

If you want the next best quantization discussion, I would look at higher-ROI knobs first:
1. calibration sample count
2. calibration dataset choice
3. whether `q_proj/k_proj/v_proj/o_proj` and MLP projections are all equally worth keeping at 4-bit
4. possibly a mixed policy where some modules stay higher precision rather than expanding 4-bit to more helper modules
