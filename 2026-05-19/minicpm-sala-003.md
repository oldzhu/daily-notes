Good questions — let me clear up several conceptual points that I conflated in the proposal.

## What "M" means

NVFP4 storage has **two** quantized things per block, not one:

1. **Per-element 4-bit code** (16 of them per block) — each is one of 16 lattice points: `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
2. **Per-block scale `s`** — one FP8 E4M3 number, applied to all 16 codes when dequantizing.

Dequant is just: `w ≈ code × s`.

`M` is **not** stored anywhere. `M` is only the **rule we use at calibration time to choose `s`**:

```
M = 6 rule:   s = max(|w_block|) / 6     → block-max maps to lattice extreme ±6
M = 4 rule:   s = max(|w_block|) / 4     → block-max maps to lattice point ±4
                                           anything > 4·s gets clipped
```

The **lattice itself never changes** (it always has 0, ±0.5, ±1, …, ±6 — that's defined by the NVFP4 standard and what the cutlass kernel decodes). Only the scaling factor `s` differs.

### Concrete example

Block of 16 weights, all small, max abs = 0.10:

- **M=6**: `s = 0.10 / 6 = 0.01667`. Step near zero is `0.5 × 0.01667 = 0.00833`. The codes ±4, ±6 of the lattice are **never used** because no weight goes that high.
- **M=4**: `s = 0.10 / 4 = 0.025`. Step near zero is `0.5 × 0.025 = 0.0125`. Wait — that's *worse*?

Hmm — let me re-check. Step near zero = `0.5 × s`. So smaller `s` ⇒ smaller step ⇒ better resolution near zero. M=6 gives smaller `s` for the same `max`. That contradicts what I said in the proposal.

Let me re-derive carefully. The benefit of M=4 is **not** finer near-zero resolution. The benefit is the opposite case:

Block of 16 weights, max abs = 0.10, but **most values are large** (clustered around ±0.08, ±0.10):

- **M=6**: `s = 0.10/6 = 0.01667`. The block-max gets coded as `6` (rounded). Mid values like 0.08 get coded as `0.08/0.01667 = 4.8 → 5? but 5 not in lattice → rounds to nearest of {4, 6} = 4 (since |0.08−4·s|=0.0133, |0.08−6·s|=0.020)`. So 0.08 → 4·s = 0.0667. Error 0.013.
- **M=4**: `s = 0.10/4 = 0.025`. Block-max codes as `4`. Mid value 0.08 → `0.08/0.025 = 3.2 → 3`. So 0.08 → 3·s = 0.075. Error 0.005. **Better.**

So I had the intuition inverted. **M=4 helps when the block is "dense at the top"** (lots of values near max), because it spends lattice points 1, 2, 3, 4 inside the active range instead of stretching to 6 (and wasting points 5, 6). **M=6 helps when the block has one big outlier** and most values are tiny — keeping s small preserves near-zero resolution.

The decision rule (per-block MSE) handles this correctly automatically — but my proposal text gave the wrong intuition for *why*. I'll fix the doc.

## So: B1 vs B2 — same effect, different where

**Yes, identical effect on the saved checkpoint.** Both produce standard NVFP4 storage where, for each block, the FP8 E4M3 scale was chosen by the FourOverSix rule instead of always-M=6.

Difference is purely *where in the call stack* the choice happens:

| | B1 | B2 |
|---|---|---|
| When | Inside modelopt's `mtq.quantize` call, replacing its scale-search | After `mtq.quantize` finishes, before `export_hf_checkpoint` |
| What we touch | modelopt's `NVFP4QTensor._quantize` (or equivalent) — subclass + monkey-patch | Just walk `model.modules()`, read each weight quantizer's stored scale + amax + original weight, compute the better scale, write it back |
| Modelopt API surface | Internal (private, can change) | Public-ish (the saved attributes are what `export_hf_checkpoint` reads) |
| Calibration data needed | None — FourOverSix is purely weight-driven (no activations needed) | None |

Since FourOverSix is **purely a weight-side decision** (it only needs `max(|w_block|)`, not activations), there's no calibration data benefit to running it inside modelopt's loop. That makes B2 strictly equal in quality and cleaner in code surface.

## Your specific question: "2 4-bit weights encoded into FP8 E4M3"?

That's a separate thing — NVFP4 packs **two 4-bit codes per uint8 byte** (so a (N, K) weight becomes a uint8 tensor of shape (N, K/2)). The FP8 E4M3 is **only** the *scale* (1 per 16-element block, so shape (N, K/16)). They're different storage:

```
weights:    uint8 packed (4-bit × 2 per byte) → asserts in modelopt_quant.py:
              assert layer.weight.dtype == torch.uint8
scales:     float8_e4m3fn (one per 16-elem block) → assert:
              assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
```

So when I said "rewrite the per-block FP8 E4M3 scale", I meant: edit only the **scale tensor** (the `(N, K/16)` float8_e4m3fn one). The 4-bit weight codes also get re-derived (because the new scale changes which lattice point each weight rounds to), but they stay 4-bit packed in uint8 the same as before. There's no nesting like "4-bit inside FP8".

## Updated TL;DR

- M is just two different **calibration rules** for picking the per-block scale `s`. Lattice and storage are unchanged.
- FourOverSix = pick the better rule per block by reconstruction MSE.
- B2 = run modelopt unmodified, then re-pick scales (and re-derive codes) before export. Same final checkpoint as B1, simpler code.
- The "FP8 E4M3" part is the *scale*, not the weights. Weights stay 4-bit packed in uint8.

Want me to:
1. **Fix the M=4 vs M=6 intuition** in the proposal doc (current text is misleading about which case helps which way), then proceed with B2; or
2. Keep the proposal as-is functionally and just have me start probing modelopt 0.43 internals on fcloud; or
3. Discuss further?
