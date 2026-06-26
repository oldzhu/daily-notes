Excellent question. Let me do a deep-dive analysis of both kernel implementations to understand the root cause of this massive performance gap.

Starting: *Read fused kernel CUDA source* (1/4)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu#100-100), lines 100 to 196

Now let me read the Marlin kernel:

Starting: *Read Marlin kernel CUDA source* (2/4)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#100-100), lines 100 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu#200-200), lines 200 to 400

Now let me find the actual Marlin kernel template — the core compute part:

Find Marlin kernel template

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin)

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1-1), lines 1 to 500

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#500-500), lines 500 to 1000

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1000-1000), lines 1000 to 1500

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1500-1500), lines 1500 to 1630

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/dequant.h#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_dtypes.cuh)

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_00_lmI8Cbcb4ckPUO5wRMKd8575__vscode-1780183079296/content.txt#1-1), lines 1 to 100

Now let me read the Marlin kernel's main loop structure:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1-1), lines 1 to 300

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#300-300), lines 300 to 500

Now let me find the main compute loop:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#500-500), lines 500 to 700

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#700-700), lines 700 to 900

Now let me find the actual inner MMA loop:

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#900-900), lines 900 to 1100

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/marlin_template.h#1100-1100), lines 1100 to 1250

Now I have a thorough understanding of both kernels. Let me compile the analysis.

Starting: *Summarize findings* (4/4)

Here's the deep-dive analysis of why the fused W4A8 REAL kernel achieves only 12-17% utilization (35-50 TFLOPS) while Marlin achieves 80-179 TFLOPS on the same SM120 hardware.

---

## Root Cause Analysis: Fused W4A8 (FP8 QMMA) vs Marlin (BF16 MMA)

### 1. The MMA Instruction Difference

**Fused kernel** uses `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
- **Tile**: 16×8 output from 16×32 A × 8×32 B — **k=32** per MMA
- **Input precision**: FP8 `e4m3` (8-bit)
- **Theoretical peak**: 296 TFLOPS (FP8 on SM120)

**Marlin kernel** uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
- **Tile**: 16×8 output from 16×16 A × 8×16 B — **k=16** per MMA
- **Input precision**: BF16 (16-bit)
- **Theoretical peak**: 148 TFLOPS (BF16 on SM120)

**At first glance, FP8 should be 2× faster** — but it's not, because of what *surrounds* the MMA.

### 2. The Dequantization Bottleneck (THE #1 KILLER)

**Fused kernel**: Does **INT4→FP8 dequantization in shared memory**, inside the inner K loop, for EVERY tile:

```cuda
// Inside K loop (kb = 0..K, step kTileK=64):
for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
    // Per thread: read int4 from global, convert to float, multiply by scale,
    // convert to fp8, write to shared memory
    int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;  // INT4 extract
    float fv = ((float)w4 - (float)z4) * __bfloat162float(scales[gid * N + ng]);
    val = __nv_fp8_e4m3(fv);  // FP32→FP8 conversion (rounding + NaN handling)
    W_fp8[n * kTileK + k] = val;  // Write to SMEM
}
```

This is **128 threads × 16,384 SMEM writes** per K-tile iteration — all serialized through shared memory bandwidth. The INT4→FP32→FP8 conversion path involves:
- **Two int→float conversions** (w4 and z4)
- **One BF16→float conversion** (scale)
- **One float subtraction**
- **One float multiply**
- **One float→FP8 rounding conversion** (expensive — requires checking for NaN/Inf, overflow, underflow)

**Marlin kernel**: Does **INT4→BF16 dequantization in registers**, fused with the MMA pipeline:

```cuda
// Inside inner loop (per 16×16 K-tile):
dequant_data(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
// Uses LOP3 (3-input logic op) + __hfma2 — NO FP32 conversion
```

Marlin's dequant uses the `lop3` instruction — a single-cycle bit manipulation that converts INT4 → FP16 in **register space** without going through shared memory or FP32:

```cuda
// lop3: extract 4-bit values and bias exponent in ONE instruction
int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
// Then: frag_b[0] = __hsub2(lo, SUB);  // FP16 subtract
//       frag_b[1] = __hfma2(hi, MUL, ADD);  // FP16 multiply-add
```

**Key difference**: Marlin keeps everything in FP16 registers using half-precision arithmetic. The fused kernel goes FP32→FP8 which is fundamentally more expensive.

### 3. Shared Memory Bandwidth Wall

**Fused kernel**: Every K-tile iteration (64 K elements) must:
1. Write `128×64 = 8192` FP8 values to SMEM for weights (16 KB)
2. Write `128×64 = 8192` FP8 values to SMEM for activations (16 KB)
3. Read them back for MMA

Total: **32 KB SMEM traffic per K-tile iteration**. With K=4096, that's `4096/64 = 64` iterations → **2 MB of SMEM traffic** per threadblock.

**Marlin kernel**: 
- Weights are pre-repacked into Marlin format at load time
- Uses `cp_async` (async copy) to stream data from global → shared via TMA-like pipeline
- **No dequant in SMEM** — dequant happens in registers
- Activations are loaded via `ldmatrix` (tensor-core-aware load) directly into register fragments
- The async pipeline overlaps global memory loads with computation

### 4. Tile Size Mismatch

**Fused kernel**: Fixed tile `M=128, N=128, K=64`. Only 4 warps, each processes `32×128` output.

**Marlin kernel**: Configurable tile sizes. On SM120, the auto-scorer selected:
```
M=1024 N=32768 K=4096 thread_m_blocks=4 thread_n=64 thread_k=128 num_threads=128
```
This means **Marlin's threadblock covers M=1024 or M=64** (depending on batch), and **each of 128 threads processes a much larger N slice**. Marlin can use up to **256 threads** per threadblock, giving better occupancy.

### 5. The Async Pipeline Advantage

Marlin uses a **multi-stage async copy pipeline** (`cp_async`) that overlaps global memory loads with computation:

```cuda
fetch_to_shared(pipe, a_off);  // Async: load next tile while computing current
// ... compute with current tile ...
wait_for_stage();  // Wait for async loads to complete
fetch_to_registers(k, pipe);  // Copy from SMEM to registers
matmul(k);  // Tensor core compute
```

The fused kernel does **synchronous loads** — every thread reads global memory directly in the inner loop, with no overlap between memory and compute.

### 6. SM120-Specific Considerations

SM120 has **warp-level MMA** (not warpgroup), which both kernels use. However:

- **Marlin's BF16 MMA** (148 TFLOPS peak) achieves **80-179 TFLOPS** (54-121% utilization) — it's actually **hitting near peak** for BF16.
- **Fused kernel's FP8 MMA** (296 TFLOPS peak) achieves only **35-50 TFLOPS** (12-17% utilization).

The FP8 MMA is not the bottleneck — **the dequantization and SMEM plumbing is**. The fused kernel spends most of its time doing INT4→FP32→FP8 conversion in SMEM, leaving the tensor cores idle.

### 7. Summary Table

| Aspect | Fused W4A8 | Marlin | Winner |
|--------|-----------|--------|--------|
| MMA instruction | `m16n8k32` FP8 | `m16n8k16` BF16 | Fused (wider K) |
| Dequant location | **Shared memory** (SMEM) | **Registers** | **Marlin** |
| Dequant precision | FP32 → FP8 | **FP16** (half2) | **Marlin** |
| Dequant instructions | ~10 (int→float, sub, mul, float→fp8) | **~3** (lop3, hsub2, hfma2) | **Marlin** |
| SMEM traffic per K-tile | **32 KB** (weights + activations) | ~8 KB (activations only) | **Marlin** |
| Memory pipeline | Synchronous | **Async cp_async** | **Marlin** |
| Tile flexibility | Fixed 128×128×64 | **Configurable** (auto-scored) | **Marlin** |
| Achieved TFLOPS | 35-50 (12-17%) | **80-179 (54-121%)** | **Marlin** |

### The fundamental lesson

The fused kernel's INT4→FP8 dequant in shared memory is a **design flaw** for this use case. The conversion overhead (FP32 arithmetic + SMEM round-trip) completely dominates the runtime, negating the 2× theoretical FP8 throughput advantage. Marlin's approach of keeping everything in FP16 registers with LOP3 bit manipulation is vastly more efficient — the dequant is essentially free.

To make a competitive fused W4A8 kernel, the INT4→FP8 dequant would need to happen **in registers** (like Marlin does), using a similar LOP3-based approach to convert INT4 directly to FP8 without going through FP32. This is a non-trivial kernel engineering task.

Completed: *Summarize findings* (4/4)



I've completed the deep-dive analysis. The core finding is that the fused kernel's INT4→FP8 dequantization path (FP32 arithmetic + shared memory round-trip) is the bottleneck — not the FP8 tensor cores themselves. Marlin's LOP3-based register-only dequant is vastly more efficient.

What direction would you like to explore next?
