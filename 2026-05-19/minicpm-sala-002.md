Yes — Phase B (FourOverSix) is the right next step. Recap of why and what it entails:

**Why now**
- Phase A proved plumbing + kernel are healthy (4× GEMM speedup, FP4 cutlass dispatched).
- The only blocker is **quality collapse on uniform NVFP4** — exactly what the champion blog warned about. FourOverSix is the champion's documented fix.

**What FourOverSix does**
- Per NVFP4 block of 16 elements, choose between two scaling modes:
  - **M=6 mode** (default NVFP4): scale chosen so the block's max maps to 6 → wider range, lower precision near zero
  - **M=4 mode**: scale chosen so max maps to 4 → narrower range, higher precision near zero
- Decision is per-block, made at quantization time based on which mode minimizes block-level reconstruction error.
- Champion observed ~40–43% of blocks pick M=4. Restores accuracy to the ≥99% band while keeping NVFP4 storage layout (so SM120 FP4 cutlass kernel still applies — no kernel work needed).

**Implementation outline (Phase B proposal — needs your approval before coding)**
1. **Calibration-time only** — purely changes how scales are computed inside `preprocess_model.py`'s `run_nvfp4_quantization`. No sglang/runtime change.
2. Replace stock `mtq.NVFP4_DEFAULT_CFG` with a custom config that overrides the per-block scale-search to try both M=6 and M=4 and pick the lower-MSE one.
3. Two viable code paths:
   - **(B1) Override modelopt's NVFP4 quantizer** — subclass `NVFP4QTensor` (or its `_quantize` static method) to do the dual-mode search. Cleaner but tied to modelopt internals.
   - **(B2) Custom `forward_loop` + post-hoc scale adjustment** — let modelopt do default NVFP4, then walk every quantizer's `amax` / scales and rewrite the per-block scale tensor with the FourOverSix decision. Less invasive, easier to roll back.
4. Same exclude list as Phase A (`lm_head`, `o_gate`, `z_proj`, `norm`, `embed_tokens`).
5. Same 90-sample stratified calibration set.
6. Validation: smoke test (single short prompt — answer should be coherent, not "unedoc.com/abc"); then full accuracy eval; then S1/S8/Smax.

**Risks**
- Modelopt 0.43's NVFP4 internal API is not stable — (B1) might break on minor version bumps. (B2) is safer.
- If after FourOverSix accuracy still collapses on long context (cwe / niah), we may need to selectively keep some layers in BF16 / FP8 (sensitivity-analysis pass).
- Champion's blog mentions FourOverSix gives them ~99% normalized; we have no guarantee on our exact MiniCPM-SALA-90 finetune — must measure.

**Alternatives to consider before approving B**
- **Quick win first?** We have v22 baseline at #22 with score 30.04, top-5 is ≥79.55. Even a perfect FourOverSix Phase B is at best a `4×` GEMM × 0.96 correctness = ~3.8× scoring uplift on the FP4 path, but only if S1/S8/Smax all benefit. Worth confirming before sinking time:
  - Phase B effort (FourOverSix calibration + debug): meaningful (2–3 iterations).
  - Phase B+Medusa (champion's full stack) is even more.
- One leaner intermediate: **mixed precision per-layer** — keep FP4 for ffn (where 4× × big GEMM matters most) and keep GPTQ-W4A8 for attention. Smaller blast radius than FourOverSix and lets us validate the speed gain shows up in the harness before committing to rewriting the quantizer.

Which path do you want to approve?
1. **Phase B (FourOverSix)** as designed — I'll write the proposal doc + code design, you approve, then I implement.
2. **Phase B-lite (mixed-precision: ffn=NVFP4, attn=GPTQ-W4A8)** first as a cheaper sanity check.
3. **Other** — discuss further.
