You're absolutely right to question this. I was wrong — let me correct:

## Path A has NO bandwidth advantage over Marlin

Both kernels do exactly the same thing:
- **Load INT4 weights** (4-bit) from HBM
- **Dequant INT4→FP16** in registers/SMEM
- **Tensor core MMA** at 148 TFLOPS

The only difference is the MMA instruction type (INT4 vs FP16), but both achieve the same 148 TFLOPS on SM120. So Path A ≈ Marlin in both bandwidth and compute.

## Corrected comparison

| Metric | Marlin | Path A | Path B |
|--------|--------|--------|--------|
| Weight HBM load | INT4 (4-bit) | INT4 (4-bit) | INT4 (4-bit) |
| Dequant | INT4→FP16 | INT4→FP16 | INT4→FP8 |
| MMA | INT4 MMA | FP16 MMA | **FP8 QMMA** |
| TFLOPS | 148 | 148 | **296** |
| vs Marlin | 1× (baseline) | ~1× | **~2×** |

## So what's the point of Path A?

Path A's value is NOT beating Marlin — it's **proving the architecture works**:
1. Validates the fused dequant+GEMM concept end-to-end
2. Debugs the tiling logic, indexing, SMEM layout (all reusable for Path B)
3. Once stable, we swap `__float2half` → `__nv_fp8_e4m3` and `wmma::mma_sync` → `tcgen05.mma` → **free 2× compute**

Path A is a correctness/plumbing milestone. The real speed win is Path B. With that cleared up — ready to continue debugging the MMA tiling bug?
