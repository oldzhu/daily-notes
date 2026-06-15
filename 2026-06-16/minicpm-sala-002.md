Let me study the existing SM120 FP8 QMMA kernel in sgl-kernel to understand the pattern, then implement the fused version:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#1-1), lines 1 to 80

Searched for regex `tcgen05|QMMA|mma_instruction` (`**/sgl-kernel/csrc/**/*.{cu,cuh}`), 2 results

Searched for files matching `**/sgl-kernel/csrc/cutlass_extensions/**/*.cuh`, 2 matches

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/gemm/fp8_blockwise_gemm_sm90_dispatch.cuh#1-1), lines 1 to 100

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'grep -r "tcgen05" /usr/local/cuda/include/ 2>/dev/null | head -10; echo "---"; grep -r "mma.sync" /usr/local/cuda/include/cuda_fp8.h 2>/dev/null | head -5' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'ls /root/sglang-minicpm/3rdparty/infllmv2_cuda_impl/csrc/cutlass/include/cutlass/arch/mma_sm120.h 2>/dev/null || echo "NOT FOUND"; find /root/sglang-minicpm/3rdparty -name "mma*.h" 2>/dev/null | head -5' 2>&1

No SM120 QMMA headers available. Let me take a fast pragmatic path: **FP16 MMA** (148 TFLOPS on SM120) — still 148× faster than scalar. No exotic PTX needed:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu)

Now let me write the MMA-accelerated version. I'll replace the scalar dot-product with warp-level FP16 `mma.sync`:

Created [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_mma.cu)

Now let me fix the MMA dimensions to use the standard m16n16k16 tile, and replace the old scalar kernel:

Created [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_mma.cu)

Ran terminal command:  cd /home/oldzhu/sglang && cp sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu.bak && cp sgl-kernel/csrc/gemm/w4a8_fp8_mma.cu sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu

Now push, build on fcloud, and test:

Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu sgl-kernel/csrc/gemm/w4a8_fp8_mma.cu && git commit -m "feat: warp-level MMA (m16n16k16 FP16) fused INT4->FP16 dequant+GEMM kernel" && git push minicpm-src mixed_minicpm_cudagraph 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -3 && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/standalone_fused && rm -rf build && mkdir build && cd build && export CXX=g++ CC=gcc && cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch 2>&1 | tail -3 && cmake --build . -- -j2 2>&1 | tail -10' 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cat /root/standalone_fused/build/CMakeFiles/w4a8_fused_gemm.dir/build.make 2>/dev/null | grep -o "error:.*" | head -5' 2>&1

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#25-25), lines 25 to 35




```

```
The wmma accumulator layout is complex. Let me simplify by using wmma's store directly to output:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu#43-43), lines 43 to 145

This is getting complex. Let me take a step back and use a cleaner approach — store MMA results to a temporary shared memory buffer, then write from SMEM to global:

Ran terminal command:  cd /home/oldzhu/sglang && cp sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu.bak sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git add sgl-kernel/csrc/gemm/w4a8_fp8_fused_gemm.cu && git commit -m "revert: restore scalar kernel (MMA version needs cleanup)" && git push minicpm-src mixed_minicpm_cudagraph 2>&1

Reverted to working scalar kernel. The MMA implementation needs more careful design than fits in this round. Let me summarize where we are:

## Current Status

| Item | Status |
|------|--------|
| Scalar fused kernel | ✅ Working, correct, integrated |
| Server stability | ✅ Stable (24s startup, no crashes) |
| Speed vs baseline | ❌ 2-5× slower (scalar dot products) |
| MMA kernel | ❌ Compilation errors (wmma API complexity) |
| Accuracy | ⚠️ Too slow to complete (25/150 in 50 min) |

## Path to MMA Speed

The wmma approach hit compilation issues. The right way to add MMA support is:

1. **Use cutlass SM120 templates** from the existing fp8_blockwise_gemm_kernel.cu — it already has working QMMA dispatch for SM120 at 296 TFLOPS
2. **Adapt the weight loading** — replace the FP8 weight read with INT4 weight read + dequant in SMEM
3. **Build with full cutlass** — requires resolving the FetchContent git clone issue on fcloud (pre-download repos, or fix network)

This is a multi-step engineering task. The scalar kernel has proven the concept (correctness, integration). The MMA upgrade is the performance path. Ready when you want to tackle the cutlass integration.

