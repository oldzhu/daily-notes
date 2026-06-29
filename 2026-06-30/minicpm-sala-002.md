Let me read the proposal doc and the existing manual kernel to plan the implementation. 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_optimized_fused_w4a8_v2.en.md)

Now let me read the existing manual kernel to understand the current structure I'll be modifying.  

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#1-1), lines 1 to 120

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#120-120), lines 120 to 196

Path B chosen. Before writing the kernel, let me look at the benchmark harness and the standalone CMakeLists to understand how the `.so` is built and called. 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)

Now let me find the benchmark harness to understand the op call signature and how to add a v2 variant.   

Searched for files matching `**/bench_w4a8_vs_marlin_real_weights.py`, 1 match

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py#1-1), lines 1 to 120

I have enough understanding now. The existing kernel's bottleneck is Phase 1 (full-tile INT4→FP8 dequant written to an 8KB FP8 SMEM buffer, then re-read in Phase 3). Path B eliminates that SMEM round-trip by dequanting each B-fragment directly in registers right before the MMA.

Let me create the v2 kernel, keeping the validated MMA + epilogue layout identical and only relocating the dequant into registers. 

Created [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v2.cu)

Now let me wire the v2 kernel into the standalone CMakeLists with a build option.  

Replacing 17 lines with 25 lines in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)


```

```


Completed: *Write register-only v2 kernel* (3/7)

Let me verify the kernel has no obvious syntax errors, then commit and push. 

Checked [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v2.cu), no problems found

Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v2.cu sgl-kernel/csrc/gemm/CMakeLists_standalone.txt && git status --short && git commit -q -m "Add register-only INT4 dequant W4A8 v2 kernel (Path B)

Eliminates the FP8 SMEM round-trip used by w4a8_fp8_qmma.cu (v1): stages
only compact raw INT4 weights (4KB) in SMEM and dequants each MMA
B-fragment in registers right before mma.sync. Identical tile/MMA/epilogue
layout to v1. Build via -DUSE_V2_KERNEL=ON." && echo "=== committed ===" && git log --oneline -1

Committed. Now push to `minicpm-src` (per repo rule, never `origin`). 

Ran terminal command:  cd /home/oldzhu/sglang && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Starting: *Build v2 on fcloud (needs approval)* (6/7)

Path B kernel is written, committed, and pushed. Here's the summary before the build step.

## What I built

w4a8_fp8_qmma_v2.cu — the register-only INT4 dequant kernel from the proposal. The key change vs v1:

| | v1 (w4a8_fp8_qmma.cu) | v2 (Path B) |
|---|---|---|
| Weight SMEM | 8 KB **FP8** (dequanted whole tile, written then re-read) | 4 KB **raw INT4** (compact, no dequant on store) |
| Dequant location | Phase 1: full-tile loop → FP8 SMEM | In registers, per B-fragment, right before `mma.sync` |
| SMEM FP8 round-trip | write 8 KB + read 8 KB | **eliminated** |
| MMA / epilogue / A-fragment | — | **identical (validated layout preserved)** |

I kept the same `torch.ops.w4a8_fused.w4a8_fp8_fused_gemm` op signature so the existing benchmark harness works as a drop-in. Built via `-DUSE_V2_KERNEL=ON`.

## Honest tradeoff (so we interpret results correctly)

v2 removes the SMEM round-trip but, because the warps split **M** (all 4 warps share the same weights), the register dequant runs **4× redundantly** across warps. So this is a genuine experiment: it wins only if the SMEM FP8 traffic v1 paid was costlier than 4× register dequant. I did **not** use the LOP3 INT4→BF16 trick because the GPTQ contiguous packing doesn't suit LOP3 without weight repacking — that's the larger Marlin-layout rewrite we'd do next if v2 shows promise.

## Next step needs your approval

To build + benchmark on fcloud:
1. Start the instance (`start-instance` — console JWT valid until 2026-06-10)
2. `git pull` in `/root/sglang-minicpm`, then build with `-DUSE_V2_KERNEL=ON` → `libw4a8_fused_gemm.so`
3. Run bench_w4a8_vs_marlin_real_weights.py (v2 vs Marlin vs v1) at the o_proj/down_proj/gate_up shapes
4. Pause the instance immediately after collecting output

May I start the fcloud instance and run the build + micro-benchmark?

