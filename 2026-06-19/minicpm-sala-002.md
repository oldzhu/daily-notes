Good questions — let me clarify each:

### Why registers aren't an option for tcgen05

It's a **hardware constraint**, not a choice. NVIDIA's tensor core generations work differently:

| Generation | Instruction | Accumulator | TFLOPS |
|------------|------------|-------------|--------|
| SM80+ (Ampere/Hopper) | `mma.sync` | **Registers** | 148 |
| SM90 (Hopper) | `wgmma` | **Shared memory** | 148 |
| SM120 (Blackwell) | `tcgen05.mma` | **TMEM** | **296** |

The 2× speedup (148→296 TFLOPS) comes specifically from `tcgen05.mma`, and that instruction **only** works with TMEM. There's no register-based variant of tcgen05. If we use the old `mma.sync` with registers, we're stuck at 148 TFLOPS — same as Marlin. That's what our current w4a8_fp8_fused_gemm.cu does, and why it's not faster.

### What TMA & TMEM mean

- **TMEM (Tensor Memory)**: New dedicated on-chip storage on Blackwell. ~64KB per SM, used **only** for tensor core accumulators. Separate from both registers and shared memory. Think of it as a specialized scratchpad that's physically close to the tensor cores.

- **TMA (Tensor Memory Accelerator)**: Hardware unit for async bulk data movement. Uses small "descriptors" (128-byte structs describing tensor layout) to move data between memory spaces without thread involvement. tcgen05.mma reads operands from shared memory **via TMA descriptors** — you can't just pass raw pointers.

### Our w4a8_fp8_qmma.cu — what's wrong

Yes, I created it, but it **doesn't compile**. The inline PTX I wrote doesn't match what `ptxas` (CUDA 12.8) expects:

```
Error: Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
Error: Feature '.cta_group::1' not supported  
Error: elect_one.sync syntax error at '0xFFFFFFFF'
```

The correct syntax (from NVIDIA's own cutlass source) is different — order of qualifiers, TMEM alloc writes to shared memory not registers, etc. The kernel has the right **structure** but wrong **PTX syntax** throughout.

### Path forward

The cutlass library already handles all of this correctly (TMA descriptors, TMEM management, correct PTX). Our `fp8_blockwise_scaled_mm` (which uses cutlass SM120 QMMA) compiles and works. The open task is creating an SM120 version of the **mixed-input** builder that handles INT4→FP8 in the mainloop — adapting the SM90 w4a8_grouped_mm_c3x.cuh pattern to SM120's tcgen05.

Want me to pursue that, or do you want to explore another direction?
