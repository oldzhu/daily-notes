Yes, Path B would significantly outperform Path A. Here's the math:

## Path A vs Path B comparison

| Metric | Marlin Baseline | Path A (FP16 MMA) | Path B (FP8 QMMA) |
|--------|----------------|-------------------|-------------------|
| **Weight storage** | INT4 (4-bit) | INT4 (4-bit) | INT4 (4-bit) |
| **Weight HBM load** | 4-bit | 4-bit | 4-bit |
| **Dequant precision** | INT4→FP16 | INT4→FP16 | INT4→FP8 |
| **GEMM compute** | INT4 MMA | FP16 MMA | **FP8 QMMA** |
| **TFLOPS on SM120** | 148 | 148 | **296** |
| **vs Marlin** | 1× | ~1.5× (bandwidth win) | **~2-3×** (bandwidth + compute) |

Path B wins because SM120's `tcgen05` FP8 QMMA delivers **296 TFLOPS** — double the FP16/INT4 tensor core rate. Combined with the 2× weight bandwidth reduction, Path B could reach **2-3× faster than Marlin baseline**.

## Why Path A first

Path A uses standard `nvcuda::wmma` (FP16 MMA) which:
- Works on any SM70+ GPU — no SM120-specific PTX needed
- Has full compiler/header support in CUDA 12.x
- Is a smaller debugging surface

Once Path A works and proves the fused dequant+GEMM concept at speed, we can upgrade to Path B by swapping FP16→FP8 in the dequant and using `tcgen05` PTX (or cutlass SM120 templates) for the MMA. The tiling logic (which we're debugging now) is the same for both.

**So: fix Path A first → validate speed gain → then Path B for the 2× compute bump.** Want me to continue debugging the MMA tiling bug?
