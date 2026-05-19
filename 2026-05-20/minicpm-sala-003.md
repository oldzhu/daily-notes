Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py#1053-1053), lines 1053 to 1245

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "import flashinfer, os; print(os.path.dirname(flashinfer.__file__))" 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "import modelopt; print(modelopt.__file__)" 2>&1 | tail -3 ; pip show nvidia-modelopt 2>&1 | head -3



Fetched https://raw.githubusercontent.com/NVIDIA/TensorRT-Model-Optimizer/release/0.43/modelopt/torch/quantization/qtensor/nvfp4_tensor.py



Fetched https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py



Fetched https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py

Got everything I need. Modelopt source confirms the exact formulas. Final design:

**Key facts from modelopt 0.43 source (`NVFP4QTensor`):**

| Item | Detail |
|---|---|
| Lattice values | `[0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]` (sign in high bit) |
| Decision boundaries | `[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]` (searchsorted) |
| Round-to-even | At odd-indexed bounds `[0.75, 1.75, 2.5]`, add +1 |
| Packing | `packed = (q[1::2] << 4) | q[0::2]` — element 0 low nibble, element 1 high nibble |
| `weight_scale_2` (float32 scalar) | `global_amax / (6 × 448)` (448 = FP8 E4M3 max) |
| `weight_scale` (FP8 E4M3 per-block) | `per_block_amax / (6 × weight_scale_2)` cast to FP8 E4M3 |
| Effective per-block dequant scale | `weight_scale.float() × weight_scale_2` |
| sglang on-disk format | `weight` uint8 `(N, K/2)`, `weight_scale` FP8 E4M3 `(N, K/16)`, `weight_scale_2` float32 scalar, `input_scale` float32 scalar |

**FourOverSix only changes `weight_scale`. `weight_scale_2` and `input_scale` stay identical to default.**

**Implementation strategy — runtime monkey-patch (cleanest of all options):**

Modelopt's `NVFP4QTensor.quantize` is structured so per-block scales come from `get_weights_scaling_factor(input, block_size, scaling_factor_2)`. If we replace that classmethod with a FourOverSix version, the rest of modelopt's `quantize` (and downstream `export_hf_checkpoint`) packs codes automatically with our chosen scales. Zero patching of pack/lattice/sf2 logic.

```python
# Pseudocode for the monkey-patched method
@classmethod
def get_weights_scaling_factor_fos(cls, input, block_size, scaling_factor_2=None, keep_high_precision=False):
    if scaling_factor_2 is None:
        scaling_factor_2 = cls.get_weights_scaling_factor_2(input)
    
    # Standard M=6 path (modelopt default)
    Wb = input.view(*input.shape[:-1], -1, block_size).float()
    per_block_amax = Wb.abs().amax(dim=-1)
    scale_unbounded = per_block_amax / (6.0 * scaling_factor_2.to(per_block_amax.device))
    scale_unbounded = torch.where(scale_unbounded == 0, torch.ones_like(scale_unbounded), scale_unbounded)
    scale_m6_fp8 = scale_unbounded.to(torch.float8_e4m3fn)
    
    # Champion's form: M=4 derived from M=6 (preserves bit-exactness for M=6 winners)
    scale_m4_fp8 = (scale_m6_fp8.float() * 1.5).to(torch.float8_e4m3fn)
    
    # MSE comparison
    sf2 = scaling_factor_2.to(per_block_amax.device).float()
    def dequant_err(scale_fp8):
        eff = (scale_fp8.float() * sf2).unsqueeze(-1).clamp_min(1e-30)
        codes = cls._cast_fp4(Wb / eff)            # reuse modelopt's lattice rounding
        # decode codes → values via e2m1_values lookup (also modelopt's helper)
        decoded = cls.get_e2m1_values(Wb.device)[codes.long()]
        return ((decoded * eff - Wb) ** 2).mean(dim=-1)
    
    err_m6 = dequant_err(scale_m6_fp8)
    err_m4 = dequant_err(scale_m4_fp8)
    pick_m4 = err_m4 < err_m6
    
    # Save stats for diagnostics
    if not hasattr(cls, "_fos_stats"):
        cls._fos_stats = []
    cls._fos_stats.append({
        "shape": tuple(input.shape),
        "n_blocks": pick_m4.numel(),
        "n_pick_m4": int(pick_m4.sum().item()),
    })
    
    final_fp8 = torch.where(pick_m4, scale_m4_fp8.float(), scale_m6_fp8.float()).to(torch.float8_e4m3fn)
    if keep_high_precision:
        return final_fp8.float(), scaling_factor_2
    return final_fp8, scaling_factor_2
```

**Confidence checks:**
- Bit-identical to Phase A on M=6 winners (we keep modelopt's exact `(amax/6/sf2).to(fp8)` value)
- Same lattice rounding (`cls._cast_fp4` reused, not reimplemented)
- Same packing (modelopt's `quantize` does the packing after the scale call)
- Stats logged → we can verify the ~40–43% M=4 pick rate champion reported

**What I'll add to preprocess_model.py:**
1. `_install_four_over_six_monkey_patch()` — saves original `NVFP4QTensor.get_weights_scaling_factor`, installs FOS variant. Returns a teardown closure.
2. `_uninstall_four_over_six_monkey_patch(restore)` — restore.
3. In `run_nvfp4_quantization`, gate by `os.environ.get("SOAR_NVFP4_FOUR_OVER_SIX")=="1"`. Install before `mtq.quantize`, uninstall after `export_hf_checkpoint`. Log stats.
4. Local unit test using a synthetic tensor (no modelopt needed if we write a tiny standalone copy of the algorithm — but I'll do the test on fcloud instead, since it requires modelopt anyway).

**What I'll add to `prepare_env.sh`:**
- In the existing `nvfp4_fos` profile branch: `export SOAR_NVFP4_FOUR_OVER_SIX=1`. Already validated by the case-statement.

**Note on the patching of `get_weights_scaling_factor`:**
- The probe showed weight_quantizer has `_amax` (scalar) but NOT `global_amax`, so the **dynamic** path is taken. This is the function I patch.
- I'll also patch `get_weights_scaling_factor_from_quantizer` (static path) defensively, in case some layers in MiniCPM-SALA use it.

Ready to implement. Want me to:
1. **Proceed with implementation** as designed.
2. Adjust the design first (specify alternative algorithm form, or different gating, etc.)
