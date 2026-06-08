Understood. We'll write/edit code locally, sync to fcloud for kernel build and testing. 

Before I start proposing code changes, let me first do a thorough reconnaissance of the existing codebase — the Marlin kernel we need to modify, the old W4A8 plumbing we can reuse, and the sgl-kernel build system. This will ground the implementation proposal.

Starting: *Research existing Marlin kernel + old W4A8 plumbing* (1/5)

Find Marlin kernel and W4A8 plumbing files

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4A8_REAL_001.en.md#60-60), lines 60 to 130

Searched for files matching `**/sgl-kernel/csrc/**/*.cu`, 70 matches

Searched for regex `SOAR_W4A8_FP8_GEMM|w4a8` (`**/python/sglang/srt/layers/quantization/**`), 20 results

Searched for files matching `**/sgl-kernel/CMakeLists.txt`, 1 match

Searched for regex `cutlass_w8a8_fp8|fp8_blockwise|fp8_gemm` (`**/sgl-kernel/csrc/**`), 20 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#690-690), lines 690 to 820

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#1-1), lines 1 to 120

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu#1-1), lines 1 to 150

Searched for regex `_soar_w4a8_eligible|w4a8` (`**/python/sglang/srt/models/minicpm_sala.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#796-796), lines 796 to 900

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#50-50), lines 50 to 140

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/qserve_w4a8_per_group_gemm.cu#1-1), lines 1 to 80

Searched for files matching `**/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py`, 1 match

Searched for files matching `**/python/sglang/srt/layers/quantization/fp8_utils.py`, 2 matches

Searched for regex `__global__|template|launch` (`**/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`), 9 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#1-1), lines 1 to 1

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py#1-1), lines 1 to 80

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_utils.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#100-100), lines 100 to 250

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/utils_w4a8_fp8.py#70-70), lines 70 to 150

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#100-100), lines 100 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#198-198), lines 198 to 380

Read [](file:///home/oldzhu/sglang/sgl-kernel/CMakeLists.txt#250-250), lines 250 to 600

Read memory

Good. The Explore agent found the complete architecture. Let me now read the key kernel file to understand the SM120 FP8 path we can build on.

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#200-200), lines 200 to 380

Now I have a complete picture. Let me present the implementation proposal.

---

## True W4A8 Implementation Proposal — Summary

After analyzing the codebase, here's the concrete plan:

### Key finding: SM120 FP8 kernel already exists

fp8_blockwise_gemm_kernel.cu has a production SM120 FP8 blockwise GEMM at lines 206-380:
- 128×128×128 MMA tiles, FP8 e4m3 inputs, BF16 output
- `Sm120BlockwiseScaleConfig` for per-128×128 block scaling
- 296 TFLOPS FP8 QMMA on SM120

**The problem with the old W4A8**: Python-side code in `gptq.py` dequantizes INT4→BF16→FP8 at load time, stores FP8 weights as persistent buffers (doubling HBM weight footprint), then feeds them to this kernel. The fix is to **keep INT4 storage and fuse the dequant into the kernel**.

### Why a kernel change is unavoidable

A naive "dequant-to-temp-FP8-then-GEMM" approach creates an intermediate FP8 tensor. For MLP layers (128 MB FP8), this hits HBM and costs 5× the weight traffic of Marlin. The dequant **must** be fused into the GEMM to avoid this round-trip.

### Proposed implementation (3-4 days)

| Day | Task | Files |
|---|---|---|
| **1** | Python-side: FP8 activation quantizer + new env gate `SOAR_W4A8_REAL_FP8_GEMM` | `gptq.py`, `minicpm.py`, `prepare_env.sh` |
| **2-3** | CUDA kernel: modify SM120 FP8 kernel to accept INT4 weights, fuse dequant into mainloop | fp8_blockwise_gemm_kernel.cu |
| **4** | fcloud build + test: accuracy + speed S1/S8/Smax | fcloud instance |

### Kernel modification approach

The SM120 kernel's `CollectiveMainloop` loads FP8 weights from HBM. We need to:
1. Change `ElementB` from `float_e4m3_t` to our INT4 packed type
2. In the mainloop's shared-memory load stage, unpack INT4→FP8 after loading into SMEM
3. Feed the dequantized FP8 values to the existing QMMA instructions
4. Keep everything else (activation path, scale handling, epilogue) unchanged

### Build and test workflow

```bash
# 1. Edit kernel locally in workspace
# 2. Sync to fcloud
python3 scripts/fcloud/fcloud_workflow.py sync

# 3. Build on fcloud (incremental)
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/sglang-minicpm/sgl-kernel && \
   export CXX=g++ CC=gcc && \
   export CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=10G && \
   make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"'

# 4. Copy wheel and test
python3 scripts/fcloud/fcloud_exec.py exec \
  'cp /root/sglang-minicpm/sgl-kernel/dist/sgl_kernel-*.whl /root/submission_sim/ && \
   cd /root/submission_sim && source prepare_env.sh'

# 5. Restart server with SOAR_W4A8_REAL_FP8_GEMM=1, run accuracy + speed
```

### Success criteria (vs v24 baseline: S1=110.54, S8=40.32, Smax=33.48, acc~77%)

| Metric | Must pass | Stretch |
|---|---|---|
| Accuracy (normalized) | ≥ 97% (C≥0.92) | ≥ 99% (C=1.0) |
| S1 | ≤ 110s (no regression) | ≤ 100s (−10%) |
| S8 | ≤ 38s (−6%) | ≤ 35s (−13%) |
| Smax | ≤ 28s (−16%) | ≤ 25s (−25%) |

### Risk: CUTLASS INT4+FP8 mixed-input

CUTLASS 3.x `CollectiveBuilder` is not designed for mixed INT4+FP8 input. We may need to write a custom mainloop rather than using the CUTLASS builder. Fallback: if the CUTLASS approach proves too complex, pivot to a hand-written CUDA kernel based on the QServe W4A8 kernel pattern (which already handles INT4 weights) but with FP8 QMMA instructions instead of INT8.

---

**Shall I start with Day 1 (Python-side FP8 activation quantizer + env gate)?** This is the lowest-risk piece and gets the plumbing ready for the kernel work.
