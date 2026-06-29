Good question first: **yes**, the numbers like `135.8` in that table are **TFLOPS** (tera-FLOP/s). The bench computes `flops = 2·M·N·K` (the 2 is multiply+add) and divides by the median kernel time: `flops / median_seconds / 1e12`. So Marlin sustaining 135.8 means ~135.8 trillion FP ops/sec — vs SM120's ~296 TFLOPS FP8 / ~148 BF16 peak. Our fused kernels at ~37 are ~12-25% utilization; that's the gap Phase 2 targets.

Now let me commit to Phase 2: the Marlin-style multi-stage `cp.async` pipeline. Let me set up the plan. 

Starting: *Design v3 cp.async pipeline kernel* (1/6)

Here's my Phase 2 design before writing it:

**Key insight from Path B:** the bottleneck is no compute/memory overlap — single-buffered SMEM with `__syncthreads` stalls between load → MMA. Phase 2 fixes this with a **3-stage `cp.async` ring buffer** (the technique Marlin uses).

**Design:**
- Triple-buffered SMEM (3 stages × ~12.3 KB = ~37 KB, under the 48 KB default).
- `cp.async` (via `<cuda_pipeline.h>`) loads raw INT4 weights + FP8 activations + scales/zeros for tile *k+2* while the MMA computes on tile *k* — overlapping HBM latency with tensor-core work.
- **Weight SMEM layout transposed to `[K-group][N]`** so the 128 columns are contiguous in both global and shared → enables 16-byte vectorized `cp.async` (impossible with v2's `[N][K-group]` layout where columns are `N`-strided).
- Register-only dequant + identical `mma.sync.m16n8k32` + epilogue from v2 (validated correct).
- Requires `N%128==0`, `K%64==0` (holds for all SALA GEMM shapes); full tiles → no per-element bounds checks, which `cp.async` needs.

Let me write it.  

Created [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v3.cu)

Starting: *Add USE_V3_KERNEL CMake option* (3/6)

Now add the v3 CMake option.   

Replacing 11 lines with 19 lines in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)


```

```


Checked [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v3.cu), no problems found

Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v3.cu sgl-kernel/csrc/gemm/CMakeLists_standalone.txt && git commit -q -m "Add cp.async pipeline W4A8 v3 kernel (Phase 2)

3-stage cp.async ring buffer prefetches tile k+2 (raw INT4 weights + FP8
acts + scales/zeros) while MMA computes tile k, overlapping HBM latency
with tensor-core work. Weight SMEM transposed to [K-group][N] for 16-byte
vectorized cp.async. MMA/dequant/epilogue identical to v2. Build via
-DUSE_V3_KERNEL=ON. Requires N%128==0, K%64==0." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Starting: *Build v3 on fcloud (needs approval)* (5/6)

The v3 cp.async pipeline kernel is written, committed, and pushed.

To answer your question fully: **yes, `135.8` = 135.8 TFLOPS**. Higher is better; SM120 peaks at ~296 TFLOPS FP8 / ~148 BF16.

## What v3 does (Phase 2)

w4a8_fp8_qmma_v3.cu — the async-pipeline rewrite:

| | v1/v2 | v3 (Phase 2) |
|---|---|---|
| SMEM buffering | single | **3-stage ring** |
| Memory↔compute | serialized (`load → sync → MMA`) | **overlapped** via `cp.async` |
| Prefetch | none | tile *k+2* loads while MMA runs tile *k* |
| Weight SMEM layout | `[N][K-group]` (scalar loads) | **`[K-group][N]`** → 16-byte vectorized `cp.async` |
| MMA / dequant / epilogue | — | identical to v2 (validated math) |

The whole point: stop the tensor cores idling during HBM loads. This is exactly the mechanism behind Marlin's 82-135 TFLOPS.

**Honest caveats** (so we read the result correctly):
- The 4× redundant register dequant across M-split warps is still present — if v3 is now memory-overlapped but compute-bound on dequant, that's the next thing to fix (dequant-once-to-SMEM).
- Correctness must be verified: the `cp.async` weight layout transpose + ring-buffer WAR hazards are the risky parts. The benchmark harness computes TFLOPS but I should add a correctness check vs a reference (Marlin output) before trusting speed numbers.

## Next: build + benchmark on fcloud (needs your approval)

Plan: start instance → `git pull` → build with `-DUSE_V3_KERNEL=ON` → run bench_w4a8_vs_marlin_real_weights.py (v3 vs v2 vs Marlin on down_proj) → **add a correctness assert vs Marlin** → pause instance.

May I start the fcloud instance and build + benchmark v3?

Made changes.Good question first: **yes**, the numbers like `135.8` in that table are **TFLOPS** (tera-FLOP/s). The bench computes `flops = 2·M·N·K` (the 2 is multiply+add) and divides by the median kernel time: `flops / median_seconds / 1e12`. So Marlin sustaining 135.8 means ~135.8 trillion FP ops/sec — vs SM120's ~296 TFLOPS FP8 / ~148 BF16 peak. Our fused kernels at ~37 are ~12-25% utilization; that's the gap Phase 2 targets.

Now let me commit to Phase 2: the Marlin-style multi-stage `cp.async` pipeline. Let me set up the plan. 

Starting: *Design v3 cp.async pipeline kernel* (1/6)

Here's my Phase 2 design before writing it:

**Key insight from Path B:** the bottleneck is no compute/memory overlap — single-buffered SMEM with `__syncthreads` stalls between load → MMA. Phase 2 fixes this with a **3-stage `cp.async` ring buffer** (the technique Marlin uses).

**Design:**
- Triple-buffered SMEM (3 stages × ~12.3 KB = ~37 KB, under the 48 KB default).
- `cp.async` (via `<cuda_pipeline.h>`) loads raw INT4 weights + FP8 activations + scales/zeros for tile *k+2* while the MMA computes on tile *k* — overlapping HBM latency with tensor-core work.
- **Weight SMEM layout transposed to `[K-group][N]`** so the 128 columns are contiguous in both global and shared → enables 16-byte vectorized `cp.async` (impossible with v2's `[N][K-group]` layout where columns are `N`-strided).
- Register-only dequant + identical `mma.sync.m16n8k32` + epilogue from v2 (validated correct).
- Requires `N%128==0`, `K%64==0` (holds for all SALA GEMM shapes); full tiles → no per-element bounds checks, which `cp.async` needs.

Let me write it.  

Created [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v3.cu)

Starting: *Add USE_V3_KERNEL CMake option* (3/6)

Now add the v3 CMake option.   

Replacing 11 lines with 19 lines in [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/CMakeLists_standalone.txt)


```

```


Checked [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v3.cu), no problems found

Ran terminal command:  cd /home/oldzhu/sglang && git add sgl-kernel/csrc/gemm/w4a8_fp8_qmma_v3.cu sgl-kernel/csrc/gemm/CMakeLists_standalone.txt && git commit -q -m "Add cp.async pipeline W4A8 v3 kernel (Phase 2)

3-stage cp.async ring buffer prefetches tile k+2 (raw INT4 weights + FP8
acts + scales/zeros) while MMA computes tile k, overlapping HBM latency
with tensor-core work. Weight SMEM transposed to [K-group][N] for 16-byte
vectorized cp.async. MMA/dequant/epilogue identical to v2. Build via
-DUSE_V3_KERNEL=ON. Requires N%128==0, K%64==0." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Starting: *Build v3 on fcloud (needs approval)* (5/6)

The v3 cp.async pipeline kernel is written, committed, and pushed.

To answer your question fully: **yes, `135.8` = 135.8 TFLOPS**. Higher is better; SM120 peaks at ~296 TFLOPS FP8 / ~148 BF16.

## What v3 does (Phase 2)

w4a8_fp8_qmma_v3.cu — the async-pipeline rewrite:

| | v1/v2 | v3 (Phase 2) |
|---|---|---|
| SMEM buffering | single | **3-stage ring** |
| Memory↔compute | serialized (`load → sync → MMA`) | **overlapped** via `cp.async` |
| Prefetch | none | tile *k+2* loads while MMA runs tile *k* |
| Weight SMEM layout | `[N][K-group]` (scalar loads) | **`[K-group][N]`** → 16-byte vectorized `cp.async` |
| MMA / dequant / epilogue | — | identical to v2 (validated math) |

The whole point: stop the tensor cores idling during HBM loads. This is exactly the mechanism behind Marlin's 82-135 TFLOPS.

**Honest caveats** (so we read the result correctly):
- The 4× redundant register dequant across M-split warps is still present — if v3 is now memory-overlapped but compute-bound on dequant, that's the next thing to fix (dequant-once-to-SMEM).
- Correctness must be verified: the `cp.async` weight layout transpose + ring-buffer WAR hazards are the risky parts. The benchmark harness computes TFLOPS but I should add a correctness check vs a reference (Marlin output) before trusting speed numbers.

## Next: build + benchmark on fcloud (needs your approval)

Plan: start instance → `git pull` → build with `-DUSE_V3_KERNEL=ON` → run bench_w4a8_vs_marlin_real_weights.py (v3 vs v2 vs Marlin on down_proj) → **add a correctness assert vs Marlin** → pause instance.

May I start the fcloud instance and build + benchmark v3?

