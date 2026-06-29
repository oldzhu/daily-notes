I'll work on fixing the CUTLASS QMMA kernel. Let me first investigate the API mismatches more deeply. Let me start the instance and examine the CUTLASS headers. 

Starting: *Investigate CUTLASS API mismatch* (1/5)

Let me first examine the existing working CUTLASS kernels in the codebase to understand the correct API patterns, before touching fcloud. Let me look at the FP8 blockwise GEMM that actually works. 

Searched for files matching `**/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`, 1 match

Searched for regex `make_cute_packed_stride|GemmUniversalAdapter|GemmUniversal|ArchTag|Sm120|Sm100` (`**/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`), 20 results

There's a working SM120 FP8 blockwise GEMM. Let me read it in full to understand the correct SM120 API pattern. 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#200-200), lines 200 to 340

The working SM120 FP8 blockwise kernel reveals the correct pattern. Let me read the full custom SM120 mixed-input builder to understand what our broken kernel dispatches to.  

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm120_mixed_input_mma_builder.inl#150-150), lines 150 to 216

Let me check if CUTLASS is checked out locally so I can inspect the API without the instance running. 

Ran terminal command:  cd /home/oldzhu/sglang && find . -path ./node_modules -prune -o -name "sm100_mixed_input_umma_builder.inl" -print 2>/dev/null | head -5; echo "---"; ls sgl-kernel/build/_deps/ 2>/dev/null | head; echo "---local cutlass?---"; find / -name "sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp" 2>/dev/null | head -3

Compacted conversation

Checked terminal output

Killed terminal

Let me read the broken kernel in full to assess its exact current state. 

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#1-1), lines 1 to 200

Let me examine the mixed-input builder dispatch header to understand SM120 routing.  

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp)

This reveals a fundamental architectural issue. Let me verify the SM120 vs SM100 MMA facts from the hardware reference doc.  

Searched for regex `tcgen05|UMMA|warp-level|warpgroup|TMEM|warp group` (`**/docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md`), 5 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md#20-20), lines 20 to 60

I've found the root cause without needing to burn fcloud build hours. Let me update the task status. 

Completed: *Investigate CUTLASS API mismatch* (1/3)

I diagnosed the CUTLASS QMMA kernel offline (no fcloud time spent) and found the compile errors are **symptoms of a fundamental architectural mismatch**, not a simple API drift.

## Root cause

w4a8_fp8_qmma_sm120.cu is built on a false premise stated in its own header and in collective_builder_mixed_input.hpp:

> "SM120 hardware is backward-compatible with SM100. Use the NVIDIA SM100 mixed-input UMMA builder..."

This is **incorrect**. Per SM120_RTX_PRO_HARDWARE.md:

| | SM100 (datacenter B100/B200) | SM120 (RTX PRO 6000D, our GPU) |
|---|---|---|
| MMA | `tcgen05` UMMA (warpgroup, TMEM-based) | warp-level `mma.sync` (no tcgen05, no TMEM) |

The kernel sets `ArchTag = cutlass::arch::Sm100` and routes through NVIDIA's `sm100_mixed_input_umma_builder.inl`, which emits `tcgen05.mma` + TMEM instructions. Those instructions **do not exist on `sm_120a`**. So:

- The `GemmUniversal "incomplete type"` error happens because the SM100 mixed-input collective never produces a valid collective for the SM120 target → `StrideA/StrideB` are never deduced → the `make_cute_packed_stride` overload errors cascade from there.
- Even if I forced it to compile, it could not run on SM120 silicon.

## Why it can't be patched quickly

The only SM120-correct path is our custom sm120_mixed_input_mma_builder.inl (uses `rr_op_selector_sm120`, the warp-level path). But two blockers remain:

1. The kernel calls the **standard** `CollectiveBuilder`, so it never reaches our SM120 builder anyway.
2. More importantly, that custom builder wires up the INT4/FP8 types but uses `cute::identity` transforms — **the actual INT4→FP8 dequant logic does not exist** in the SM120 warp-level mainloop. Upstream CUTLASS (commit `57e3cfb47a`) has **no GemmUniversal kernel-layer specialization for an SM120 mixed-input dequant mainloop**. Writing that mainloop is the real work — matching the 2-3 week estimate in the research doc.

So "fixing the compile errors" would not yield a working or correct kernel; it would need a new SM120 mixed-input mainloop written from scratch.

## Recommendation

I'd stop here on the CUTLASS QMMA path and choose one of:

- **A (safe, ships today):** Keep the v25 two-step path (81.07%, C=1.0, currently the working baseline) and move optimization effort elsewhere in the catalog.
- **B (highest upside, the kernel idea you originally raised):** Build the **register-only LOP3 INT4→BF16→FP8 manual kernel** from `PROPOSAL_optimized_fused_w4a8_v2.en.md`. This uses SM120's actual warp-level `mma.sync` (the correct instruction set), borrows Marlin's LOP3 dequant + `cp.async` pipeline, and is a tractable, single-file effort — unlike the CUTLASS mixed-input rabbit hole.
- **C (research):** Commit to writing a proper SM120 mixed-input CUTLASS mainloop (multi-week, high risk).

Which path do you want? I did **not** start the fcloud instance, so no billing was incurred. If you still want me to attempt forcing the CUTLASS kernel to compile just to confirm the failure empirically, I can — but it won't produce a runnable SM120 kernel.
