Yes, you're right. The goal is:

> **Implement a fused INT4→FP8 dequant + GEMM kernel that improves benchmark speed vs the Marlin baseline, without losing accuracy.**

Here's where we stand:

## Goal vs Reality

| Requirement | Status | Detail |
|-------------|--------|--------|
| **Correctness** | ✅ Scalar kernel | Bit-exact vs reference at all model dims |
| **Accuracy preservation** | ✅ Expected | FP16 dequant (more precise than FP8 ref) |
| **Speed vs Marlin** | ❌ 2-5× slower | Scalar kernel does no tensor core work |
| **Server integration** | ✅ Done | gptq.py priority swap, fallback, padding |
| **MMA kernel** | ⚠️ Broken at scale | Single-tile works, multi-tile has bugs |
| **QMMA (296 TFLOPS)** | ❌ Not started | No SM120 tcgen05 PTX available |

## The gap

The Marlin baseline runs at **148 TFLOPS** (INT4 tensor cores). Our scalar kernel runs at **<1 TFLOPS** (element-wise dot products). To win, we need at least equivalent tensor core throughput.

## Two paths to goal

| Path | Effort | Risk | Speed target |
|------|--------|------|-------------|
| **A: Fix MMA kernel** (m16n16k16 FP16) | Medium | Low (uses standard wmma) | 148 TFLOPS (match Marlin) + 2× bandwidth win |
| **B: Cutlass SM120 QMMA** (tcgen05 FP8) | High | High (no headers/docs) | 296 TFLOPS (2× Marlin) |

**Path A** is the pragmatic choice — fix the remaining tiling bug in the MMA kernel (error ~68 on 4096-dim matrices), and we have a kernel that matches Marlin's compute throughput PLUS saves 2× weight bandwidth. That should beat baseline speed.

Want me to focus on debugging the MMA tiling bug?
