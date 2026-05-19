Yes — that's the algorithm I'll use, and it's actually slightly better than the form I wrote in the proposal. Quick math note so we're on the same page:

**Equivalence + why champion's form is preferred:**

- My proposal form: `s_M6 = fp8(max/6)`, `s_M4 = fp8(max/4)` — two independent FP8 rounds.
- Champion form: `s_M6 = fp8(max/6)`, `s_M4 = fp8(s_M6 × 1.5)` — M=4 is derived from the already-rounded M=6.

Since `1/4 = 1.5/6`, the unrounded values are mathematically identical. The difference is **only in how the FP8 E4M3 rounding error propagates**:

- Champion form ensures **the M=6 case is bit-identical to modelopt's default NVFP4 output** (since we keep the same `s_M6` modelopt computed). Only the blocks that pick M=4 deviate from default. This makes diffing / debugging easy: blocks where M=6 wins have *zero* change vs Phase A.
- It also removes one FP8 round-trip on the rejection path (we never compute `fp8(max/4)` for M=6 winners).

**Algorithm I'll implement (with the champion's exact form):**

```python
# For each Linear's BF16 weight W of shape (N, K)
B = 16
Wb = W.view(N, K // B, B).float()                     # (N, K/B, B)
amax = Wb.abs().amax(dim=-1)                          # (N, K/B)

# M=6: standard NVFP4 (matches modelopt default)
scale_m6_raw = amax / 6.0
scale_m6 = scale_m6_raw.to(torch.float8_e4m3fn).float()  # FP8 round

# M=4 derived from M=6 (champion form)
scale_m4 = (scale_m6 * 1.5).to(torch.float8_e4m3fn).float()

# Round to NVFP4 lattice {0,±0.5,±1,±1.5,±2,±3,±4,±6}, dequant, MSE
def quant_dequant(Wb, scale):
    s = scale.unsqueeze(-1).clamp_min(1e-30)
    codes = round_to_nvfp4_lattice(Wb / s)             # exact lattice round
    return codes * s, codes

deq_m6, _   = quant_dequant(Wb, scale_m6)
deq_m4, _   = quant_dequant(Wb, scale_m4)
mse_m6 = ((Wb - deq_m6) ** 2).mean(dim=-1)            # (N, K/B)
mse_m4 = ((Wb - deq_m4) ** 2).mean(dim=-1)

pick_m4 = mse_m4 < mse_m6
final_scale = torch.where(pick_m4, scale_m4, scale_m6)  # FP8 E4M3 already

# Re-pack codes with the chosen scale
final_codes = round_to_nvfp4_lattice(Wb / final_scale.unsqueeze(-1)).to(torch.int8)
# pack two 4-bit codes per byte → uint8 (N, K/2)
packed = pack_nvfp4(final_codes)
```

A couple of details that the local research (option 1) needs to nail down before I write the final code:

1. **Exact lattice rounding**: NVFP4 codes `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` are non-uniform. I'll either reuse modelopt/flashinfer's existing rounding helper (preferred) or write a small `searchsorted`-based one. Verifying which symbol modelopt exposes is part of step 1.
2. **Pack format**: NVFP4 packs two 4-bit codes per byte. Need to confirm packing order (low-nibble = first element vs second) by reading flashinfer's unpack code or modelopt's `export_hf_checkpoint` so our packed output matches what `flashinfer.mm_fp4` decodes.
3. **Scale layout in safetensors**: probably `weight_scale` shape `(N, K/16)` dtype `float8_e4m3fn`, but might be transposed/interleaved — sglang loader's name `weight_scale_interleaved` suggests interleaving. Need to read modelopt's export code.

I'll now do option 1: read modelopt 0.43's `export_hf_checkpoint` source + sglang's `modelopt_fp4` loader to pin down all three. Want me to proceed with that research now?
