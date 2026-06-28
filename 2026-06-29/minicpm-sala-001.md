Great questions. Let me investigate both files thoroughly.

Starting: *Research w4a8_fp8_qmma_sm120.cu origin* (1/3)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu#160-160), lines 160 to 223

Now let me check the git log to see who wrote this and when:

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline --follow -20 -- sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu 2>/dev/null || echo "no git history"

Ran terminal command:  cd /home/oldzhu/sglang && git log --format="%H %an %ai %s" -1 -- sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu 2>/dev/null

Ran terminal command:  cd /home/oldzhu/sglang && git log --format="%H %an %ai %s" --reverse -- sgl-kernel/csrc/gemm/w4a8_fp8_qmma_sm120.cu 2>/dev/null | head -5

So it was written by us (oldzhu) on 2026-05-22. Let me now check if this CUTLASS QMMA kernel was ever benchmarked or used:

Searched for regex `qmma_sm120|sm120_qmma|w4a8_fp8_fused_gemm` (`**/python/sglang/srt/layers/quantization/gptq.py`), 2 results

Searched for regex `w4a8_fp8_fused_gemm|w4a8_fp8_qmma_sm120` (`**/sgl-kernel/csrc/gemm/**`), 10 results

Ran terminal command:  cd /home/oldzhu/sglang && grep -r "w4a8_fp8_qmma_sm120\|sm120_qmma" sgl-kernel/csrc/gemm/CMakeLists* 2>/dev/null || echo "not found in CMakeLists"

Searched for regex `qmma_sm120|w4a8_fp8_qmma_sm120` (`**/sgl-kernel/csrc/**/CMakeLists*`), no results

Searched for regex `qmma_sm120|w4a8_fp8_qmma_sm120` (`**/sgl-kernel/**/CMakeLists*`), no results

Ran terminal command:  cd /home/oldzhu/sglang && grep -r "qmma_sm120\|w4a8_fp8_qmma_sm120" sgl-kernel/ 2>/dev/null | head -20

So the CUTLASS QMMA kernel (w4a8_fp8_qmma_sm120.cu) is **not compiled or used anywhere** — it's dead code. Let me also check what w4a8_fp8_mma.cu (the old WMMA kernel) and the actual compiled w4a8_fp8_qmma.cu (the current kernel) are:

Ran terminal command:  cd /home/oldzhu/sglang && grep -r "w4a8_fp8_qmma\|w4a8_fp8_mma\|w4a8_fp8_fused" sgl-kernel/csrc/gemm/CMakeLists.txt 2>/dev/null | head -20

Ran terminal command:  cd /home/oldzhu/sglang && grep -rn "w4a8_fp8" sgl-kernel/csrc/ 2>/dev/null | head -30

Now let me check the CMakeLists to understand the build structure:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt#1-1), lines 1 to 60

Now I have the complete picture. Let me also quickly check the w4a8_fp8_mma.cu (old WMMA kernel) to understand the evolution:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_mma.cu#1-1), lines 1 to 50

Now let me also check how many different kernel versions exist to understand the full picture:

Ran terminal command:  cd /home/oldzhu/sglang && ls -la sgl-kernel/csrc/gemm/w4a8_fp8* 2>/dev/null

Completed: *Research w4a8_fp8_qmma_sm120.cu origin* (1/3)

Here's the full analysis answering your questions:

---

## 1. Who wrote w4a8_fp8_qmma_sm120.cu?

**We did** — all by `oldzhu` on 2026-05-22. It's a CUTLASS-based implementation that uses NVIDIA's SM100 CollectiveBuilder API (backward-compatible with SM120). It was created in 10 commits over ~5 hours.

**Crucially: it's never been compiled, benchmarked, or used.** It's not in any CMakeLists.txt — it's dead code. The only compiled kernel is w4a8_fp8_qmma.cu (the manual one we benchmarked at 35-50 TFLOPS).

The full kernel family:

| File | Approach | Status | TFLOPS |
|------|----------|--------|--------|
| w4a8_fp8_mma.cu | Old WMMA (FP16 MMA, 148 TFLOPS peak) | Replaced by QMMA | Not tested |
| w4a8_fp8_qmma.cu | **Manual FP8 QMMA** (currently compiled) | **Active** | **35-50** |
| w4a8_fp8_qmma_sm120.cu | CUTLASS SM100 QMMA | **Dead code** | **Unknown** |
| `w4a8_fp8_fused_gemm.cu` | Another manual variant | Backup | Not tested |
| `w4a8_fp8_dequant.cu` | Separate dequant pass | Not a GEMM | N/A |

## 2. CUTLASS QMMA vs Marlin — What's the difference?

**Marlin** (gptq_marlin.cu):
- **INT4 weights + BF16 activations → BF16 MMA** (148 TFLOPS peak)
- Dequant: INT4→BF16 in **registers** via LOP3 (proven, efficient)
- Async `cp_async` pipeline for memory/compute overlap
- Auto-scored tile config selection
- **Achieves: 80-179 TFLOPS (54-121% utilization)**

**CUTLASS QMMA** (w4a8_fp8_qmma_sm120.cu):
- **INT4 weights + FP8 activations → FP8 MMA** (296 TFLOPS peak)
- Dequant: INT4→FP8 handled by CUTLASS's **SM100 mixed-input builder** internally
- Uses NVIDIA's highly optimized **TMA (Tensor Memory Accelerator)** for async data movement
- Warp-specialized scheduling (separate warps for data movement vs compute)
- **Theoretical: up to 296 TFLOPS — 2× Marlin's peak**

**The key difference**: CUTLASS QMMA uses NVIDIA's proprietary `tcgen05` QMMA path (quantized MMA), which is a **different instruction** from the manual `mma.sync` we used. It's designed specifically for mixed-precision (INT4 weights + FP8 activations) and handles the dequant internally in hardware. If it works correctly, it should be **much faster** than our manual kernel.

## 3. Is GEMM optimization the right priority?

**Yes, absolutely.** Your profiling data from Round 13e showed:

| Context Length | GEMM % of GPU time |
|---------------|-------------------|
| 32K tokens | **83.6%** |
| 64K tokens | **82.8%** |
| 128K tokens | **82.4%** |

GEMM is **~83% of all GPU kernel time** at every context length. This dwarfs everything else (attention is only ~4-8%). Optimizing GEMM is the single highest-impact thing we can do.

## 4. Recommendation

The highest-priority action is to **benchmark the existing CUTLASS QMMA kernel** (w4a8_fp8_qmma_sm120.cu). Since it's already written, we just need to:

1. Add it to the CMakeLists build
2. Compile it on fcloud
3. Run the same micro-benchmark we used for the manual kernel

If it achieves **≥100 TFLOPS**, we have a working solution immediately — no need to build a new manual kernel. If it's also slow, then the FP8 QMMA path has fundamental issues on SM120 and we should invest in the register-only LOP3 approach instead.

Would you like me to start with benchmarking the CUTLASS QMMA kernel?
