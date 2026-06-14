Correct — the fused kernel currently **does NOT handle decode** (M = 1 token). Here's why:

### The constraint

The kernel uses **tile size `kTileM = 128`** — each thread block processes exactly 128 rows of the output. The host function enforces:

```cpp
TORCH_CHECK(M % kTileM == 0, "M must be multiple of 128");
```

For decode, `M = 1`, which is not a multiple of 128 → falls back to Marlin.

### Why not just pad to 128?

Padding M=1 to 128 means computing **128× more work than needed** — the kernel would multiply 127 rows of zeros against the weight matrix and then discard them. That's a 99% compute waste, which would make decode **slower** than Marlin, not faster.

### Why this is actually OK

| Phase | M (batch size) | Kernel used | Why |
|-------|---------------|-------------|-----|
| **Prefill** | 128–4096+ tokens | Fused INT4→FP8 GEMM | Saves 2× weight bandwidth |
| **Decode** | 1 token per step | Marlin (INT4 GEMM) | Already near-optimal for M=1 |

The fused kernel's value proposition is **prefill bandwidth reduction**: reading INT4 weights (4-bit) from HBM instead of FP8 weights (8-bit). During decode, memory bandwidth is dominated by KV cache access, not weight loading — so Marlin is fine.

### Competitively, this is the right trade

The official speed benchmark was updated to include **more long-context samples** (per organizer announcement). Long contexts mean:
- Large prefill token counts (M ≫ 128) → fused kernel activates
- Decode steps with large KV cache → weight bandwidth matters less

Prefill throughput is the main optimization target, and that's exactly where the fused kernel helps.

### Could we make it work for decode later?

Yes — add partial-tile handling in the kernel:
```cpp
// In output write: only write rows where mg < M
if (mg < M && ng < N)
    c_bf16[mg * ldc + ng] = __float2bfloat16(acc[n]);
```
This already exists in the kernel. The real issue is the `TORCH_CHECK` in the host function — we'd remove it and launch the grid with `ceil(M/128)` blocks instead of `M/128`. The last block would process fewer rows. But for M=1, this is still wasteful (127 idle threads ≠ compute waste, threads are free — but SMEM allocation per block is the same).

**Net: the decode fallback is by design, not a bug.**
