Short answer: **the test is feasible but the engineering effort is medium-to-high and the expected upside is small (~0.5–1 pp acc, ~no speed change).** I'd recommend skipping it and going to the GPTQ catalog. Details below.

## What "GPTQ + NVFP4 FOS + Marlin W4A16 decode" actually requires

To run this end-to-end on our stack, four pieces must all line up:

1. **GPTQ OBQ** (Hessian + error compensation) — we have this.
2. **Snap each column to FP4 E2M1 levels** instead of uniform INT4 levels during OBQ rounding — gptqmodel does NOT ship this. Need a custom `Quantizer` that, given a BF16 column + FP8 block scale, picks the nearest of `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} × block_scale` instead of nearest of `{−8…+7} × group_scale`. ~1 day of Python.
3. **Marlin kernel that consumes FP4 codes** — the existing Marlin kernel does inline arithmetic dequant: `bf16 = scale × (int4_code − zp)`. FP4 levels are **non-uniform** so a multiply‑subtract is mathematically wrong. The dequant inside marlin_template.h lines 25–60 has to be replaced with a 16‑entry FP4 LUT lookup. ~3–5 days of CUDA work + correctness tests.
4. **Scale layout decision** — Marlin assumes one FP16 scale per group=128. NVFP4 carries one FP8 scale per block=16 (8× more scales). To reuse Marlin you must either:
   - (a) drop block=16 and re-derive a single FP16 scale per group=128 from the 8 child block scales (lossy — destroys the FOS benefit), or
   - (b) extend Marlin's smem layout to load a per‑16‑element FP8 block scale on top of the per‑128 group scale (kernel rework, ~3–5 more days).

(a) is fast but **defeats the entire reason to use FP4 over INT4** — once you fold 8 block scales into one group scale you've thrown away the per‑16 dynamic range, so the result is just GPTQ-INT4-with-different-rounding-levels. Expected accuracy ≈ GPTQ-INT4 baseline ± noise.
(b) preserves the FOS benefit but is a real kernel project.

So the realistic "champion-faithful" version is **option (3) + (4b)**: ~1.5–2 engineer-weeks of focused CUDA + calibration work. The cheap "option (4a)" version is ~3 days but is unlikely to beat GPTQ baseline.

## What about FOS?

FOS picks M=4 vs M=6 **per FP8 block of 16**. That decision only has meaning if the runtime kernel actually uses the per‑block FP8 scale. So FOS only survives in option (4b), not (4a). In other words: if you go with the cheap repack, FOS becomes a no-op.

## Realistic outcome at full effort

| Metric | GPTQ baseline (our 99.11% norm) | GPTQ+FP4-FOS+Marlin (predicted) |
|---|---|---|
| Accuracy | 99.11% norm | **+0.3–1 pp** (FP4 levels track LLM weight tails better than uniform INT4 at same 4 bits, FOS adds another sliver) |
| S1 | 121.71 s | **±2%** — same Marlin kernel, same tile picker, same BF16 path |
| S8 | 44.09 s | **±2%** |
| Smax | 35.86 s | **±2%** |
| Risk | known good | new kernel + new calibration path → multiple round-trip debug cycles |

The "champion's recipe" is real and clever, but its benefit is **a small accuracy bump on top of GPTQ**, not a speed jump. They're at the top of the leaderboard because of *other* things stacked on top of it (speculative decoding heads, scheduling, KV/attention work, etc.). FP4‑levels‑in‑Marlin is the boring piece of their stack, not the decisive one.

## A cheaper "is this even worth it?" pre-test (1 day)

Before doing the kernel work, we can run a **calibration-only proxy** to upper-bound the accuracy gain:

1. Take our GPTQ-OBQ pipeline as-is.
2. Replace the rounding step with a hard snap to FP4 E2M1 levels × per-group FP16 scale (option (4a) — cheap path, no kernel changes).
3. Save as a normal GPTQ checkpoint with `bits=4`, group=128, but quantized levels are a 16‑entry FP4 subset of representable values instead of uniform INT4.
4. Load with the existing `gptq_marlin` loader — **zero kernel work**, zero sglang change.
5. Run S1 + accuracy.

This test takes **~1 day** (a small `Quantizer` subclass in our gptqmodel calibration path, plus one fcloud quant + eval round). It establishes:

- **If accuracy improves ≥ +0.5 pp** over current GPTQ baseline → the champion's recipe has merit; consider the full kernel project (option 4b).
- **If accuracy is flat or worse** → confirmed not worth the kernel work; GPTQ-INT4 + sparse_qkv_w8 is already on the right side of the FP4 vs INT4 tradeoff for this model. Park.

This is a low-cost, low-risk decision gate. Want me to write up a proposal for that 1‑day calibration-only proxy, or skip directly to the GPTQ_FP8_DENSE catalog as originally planned?
