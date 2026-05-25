Three questions, three answers:

## Q1 — Would "FP4 weights repacked into Marlin W4A16" beat our GPTQ baseline?

**Probably not, and the champion's post is consistent with that.** Read the recipe carefully: it says "GPTQ + NVFP4 + Marlin W4A16 decode". That is **three layered pieces**, not "FP4 instead of GPTQ":

1. **GPTQ** — still does the Hessian‑aware OBQ calibration on BF16 weights (the accuracy‑critical step).
2. **NVFP4 (E2M1)** — the storage format for the quantized weight (4 bits/elem, 16‑element blocks, FP8 block scale). This is a **code‑book swap** at the very end of GPTQ: instead of rounding each column to uniform INT4 levels, round to FP4 E2M1 levels.
3. **Marlin W4A16 decode** — the runtime kernel. BF16 activations, FP4 weights dequantized inline.

The win over our current GPTQ baseline (uniform INT4 levels) comes from **(2)**: FP4 E2M1's non‑uniform levels track LLM weight distributions a bit better than uniform INT4 at the same 4 bits, *if* you already paid the OBQ cost. It buys at most ~0.5–1 pp accuracy (factor 3.4 in our deep‑dive) — and **that is competing against `sparse_qkv_w8` which buys 1–2 pp by upgrading 24 sensitive linears to W8**. The two ideas are roughly the same size; whichever you pick, you eat the calibration cost once.

So the realistic outcome of porting the champion's recipe is:

- **+0.5–1 pp accuracy** (FP4 levels vs INT4 levels), maybe a hair more if FOS picks per block.
- **+0–small speed change** — Marlin kernel is the same, only the dequant LUT differs (16‑entry FP4 LUT vs 16‑entry INT4 LUT). Both are inline ops in the kernel.

Net: **incremental, not a step change.** It is **not** the source of the leaderboard gap. The champion's headline gain almost certainly comes from architectural pieces *outside* the storage code‑book — speculative heads, scheduling, KV/attention work — not from "FP4 vs INT4".

### Does the repack itself add cost vs GPTQ INT4?

For us **the repack is mostly a one‑time offline cost**, paid in preprocess_model.py. It is bookkeeping (read FP4 codes + FP8 block scales, fold into FP16 per‑group scales, write Marlin tile layout) — seconds to a minute. **Negligible** compared to OBQ calibration itself (which dominates preprocess wallclock today).

The **real** added cost is engineering: there is **no off‑the‑shelf GPTQ‑to‑FP4‑to‑Marlin path in this fork**. It needs:
- a custom GPTQ rounding step that snaps to FP4 E2M1 levels instead of INT4 (gptqmodel doesn't ship this),
- a Marlin kernel variant whose 16‑entry dequant LUT is FP4 levels, OR a scale‑folding trick that lets the existing INT4 Marlin reproduce FP4 numerics exactly.

That's a multi‑day kernel + calibration project for ~1 pp. With the 5‑hour preprocess budget and our current 56.63 → top‑5 gap, this is **not a high‑ROI target**.

## Q2 — "NVFP4 is ~9% heavier because of extra block scales, right?"

**Correct, and a small clarification:** the weights themselves are *exactly the same size* — 4 bits/elem either way (8 MB for 4096² packed). The 9% comes purely from the **scale tensor**:

| Path | weight | scale | total per 4096² layer |
|---|---|---|---|
| GPTQ INT4, group=128, FP16 scale | 8.00 MB | (32 × 4096) × 2 B = 0.25 MB | **8.25 MB** |
| NVFP4, block=16, FP8 e4m3 scale | 8.00 MB | (256 × 4096) × 1 B = 1.00 MB | **9.00 MB** |

Two factors at once: NVFP4 uses **8× more scale entries** (block 16 vs group 128) but each is **half the bytes** (FP8 vs FP16) → net 4× the scale bytes (1 MB vs 0.25 MB), which is +9% of the layer total. That extra 0.75 MB/layer × ~80 layers × 6 linears ≈ 360 MB of additional weight‑traffic per token‑step at decode — the dominant cause of the S1 slowdown we measured.

(Note: NVFP4 also stores per‑tensor `input_scale`, `weight_scale_2`, `alpha`, but those are scalars per partition — bytes negligible.)

The champion's "Marlin W4A16 decode" recipe avoids this overhead too: when you fold FP4 levels into a per‑group FP16 scale (group=128), the on‑disk scale tensor reverts to 0.25 MB. So **going via Marlin recovers the ~9% memory delta** *and* keeps the FP4 accuracy benefit. That's the part of their recipe that *is* a clear win for our path; it just doesn't exist as code today.

## Q3 — Can we add small‑M tile configs + runtime scorer to the FP4 dispatcher?

**Technically yes. Practically: it is the right idea but a non‑trivial CUTLASS project, and the upside is the same as path Q1 — only at large M.**

### Why it's harder than just dropping in new tile shapes

CUTLASS `OpClassBlockScaledTensorOp` on SM120 is **not parametrically free**. The block‑scaled MMA has fixed contracts:

- **Block scale of 16 along K** is baked into the instruction; you can't pick arbitrary K tiles.
- **Tensor `A` layout** (interleaved FP4 + interleaved FP8 scale) is rigid; the SM120 specialization in nvfp4_scaled_mm_kernels.cu only changes M tile and N tile.
- **Cluster shape 1×1×1** is the only one validated for SM120 in this code; SM100 has wider cluster options.
- **Smallest profitable M tile** for `OpClassBlockScaledTensorOp` is typically 64 (CUTLASS template constraint). Going down to M=16 like Marlin requires either falling back to a non‑block‑scaled tensor‑op kernel + software dequant, OR a bespoke epilogue.

So a realistic small‑M path looks like one of:

1. **Add a 64×N×128 tile** for `M ≤ 64` (in addition to 128/256). Not 16×N. Quick to try; modest decode help.
2. **Add a 32×N×128 tile** if CUTLASS 3.8 SM120 templates allow (uncertain — needs prototype).
3. **Below M=32**: dequant FP4→BF16 in a custom kernel and use a **BF16 GEMM** with Marlin‑style small‑batch tiles. This is exactly what "Marlin W4A16 decode" is.

### Plus the dispatcher work

A runtime scorer modeled on Marlin's gptq_marlin.cu is straightforward IF the candidate kernels exist. It picks a tile based on M, N, K, occupancy, smem fit, wave count. For FP4 we'd need to:

- enumerate the kernel variants (compile a few `Fp4GemmSm120<config, bfloat16_t>::Gemm` instantiations),
- expose them via a function table,
- write a `dispatch_sm120_small_m` cousin to `cutlass_fp4_bf16_gemm_dispatch_sm120` that scores M against {16, 32, 64, 128, 256} cutoffs.

Engineering size: **multi‑week kernel project**, not a side activity, and it only pays off **if we keep an NVFP4 path at all** — which the deep‑dive concluded we shouldn't right now.

### Why I still flag it as the right idea long‑term

If competition v2 or later removes the W4A16 BF16 ceiling (148 TFLOPS) as the binding constraint — or we move to a model where decode tokens see longer prefill chunks per step — small‑M FP4 tiles become valuable. Right now they don't beat Marlin, because Marlin's small‑M shapes are already well‑tuned and BF16 peak isn't the bottleneck at M=1–8 anyway (memory is). Building this is not on the catalog **until** memory pressure eases.

## Summary

| Question | Verdict |
|---|---|
| Does "GPTQ + FP4 levels + Marlin W4A16 decode" beat our GPTQ baseline? | Marginally (~0.5–1 pp, similar to `sparse_qkv_w8`). Not the leaderboard‑gap source. **Multi‑day** engineering for incremental gain — low ROI now. |
| Is NVFP4's ~9% heavier weight from extra block scales? | **Yes**: weights identical 8 MB; scale tensor is 1 MB (FP8 e4m3 × 256 entries) vs 0.25 MB (FP16 × 32 entries). 4× scale bytes → +9% per layer. |
| Can we add Marlin‑style small‑M tile picker to FP4 dispatcher? | **Yes in principle**, but CUTLASS `OpClassBlockScaledTensorOp` constrains tile shapes (smallest practical M ~64, not 16); going below requires a bespoke FP4→BF16 dequant kernel — which is essentially the "Marlin W4A16 decode" path again. **Multi‑week kernel project**; only worth doing if we recommit to NVFP4. |

Recommendation stays the same as the deep‑dive: **park NVFP4, pivot to GPTQ_FP8_DENSE catalog**. The champion's recipe is clever but it's a 1‑pp lever in a category where we need 30+ pp to reach top‑5.
