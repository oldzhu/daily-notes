I’m checking the upstream docs for FlyDSL, Triton, and AITER’s own backend story so I can answer this in terms of likely performance upside versus implementation risk on these three kernels.



Fetched 4 resources

Yes, but only selectively.

AITER is already a frontend over multiple backend types. Its own README says the underlying kernels can come from `triton`, `ck`, or `asm`, and it explicitly says `FusedMoE` can use optional FlyDSL-based kernels for mixed-precision MoE. So the right way to think about this is not “AITER or FlyDSL or Triton,” but:

- AITER as the operator layer
- FlyDSL/Triton as possible kernel-generation backends or custom-kernel routes underneath it

On the “generate better hardware instructions” question: also yes, in principle. Triton and FlyDSL both lower to ROCm’s compiler stack and ultimately produce AMD GPU machine code. But that does not mean they automatically beat AITER’s existing best kernels. AITER already contains hand-tuned or highly tuned kernels, including ASM paths. Generated code can beat those only if:
- the existing backend is not optimal for this exact shape/workload
- you exploit problem-specific structure better
- you tune tile sizes, memory movement, vectorization, wave occupancy, and fusion more effectively than the current implementation

For these three problems, the likely payoff is different:

1. moe-mxfp4 is the best candidate for FlyDSL exploration.
AITER explicitly mentions FlyDSL support for mixed-precision FusedMoE. That makes this the most credible place where FlyDSL could matter. If AMD told you FlyDSL is in play, this is probably what they meant. A custom or alternate FlyDSL-backed MoE kernel could plausibly change kernel selection and beat the current `fused_moe` path in moe-mxfp4/submission_clean.py if the current CK/default path is leaving performance on the table.

2. mxfp4-mm is the most plausible Triton candidate.
This problem is structurally simpler than MLA or MoE. A custom Triton matmul or quantize+matmul path is realistic. The issue is that beating AITER’s `gemm_a4w4` on MI355X is still hard, especially if AITER is already dispatching a tuned ASM kernel. Triton is worth considering here mainly if we can fuse the A-side quantization with the matmul or specialize hard for the benchmark shapes.

3. mixed-mla is the least attractive full-rewrite target.
The current path in submission_clean.py already rides AITER’s persistent MLA decode kernel. Replacing that with a custom Triton or FlyDSL kernel in hackathon time is high risk. MLA decode has complicated metadata, paged KV handling, split/reduce behavior, and persistent-kernel details. Wrapper-side gains were real; a from-scratch kernel is much more expensive and much more likely to burn time.

So the practical answer is:

- FlyDSL: potentially very useful for moe-mxfp4
- Triton: potentially useful for mxfp4-mm, maybe for auxiliary pieces of MLA, not likely for a full MLA replacement
- Neither is a guaranteed win just because they “generate instructions”
- Beating AITER’s existing tuned kernels usually requires either fusion or very workload-specific specialization

My recommendation for this repo is:

1. Do not rewrite mixed-mla in Triton/FlyDSL.
2. If we explore a new backend, start with moe-mxfp4 via FlyDSL-backed FusedMoE.
3. Backup path: explore whether mxfp4-mm can fuse A-side quantization with GEMM in Triton or a lower-level AMD DSL path.

If you want, I can take the next step and do a focused research pass on one of these:
1. FlyDSL feasibility for moe-mxfp4
2. Triton feasibility for mxfp4-mm
3. whether AITER in this environment is already capable of using FlyDSL for MoE and how to force that path
