Good question. Let me analyze the impact:

## `.clone()` Cost Analysis

| Aspect | Impact | Why |
|--------|--------|-----|
| **Memory** | Negligible | GEMM output is `[batch, hidden_dim]`. At max batch=24, hidden=4096: 24×4096×2 bytes = **~192KB** temporary. Freed immediately after clone. |
| **Speed** | Negligible | A 192KB cudaMemcpy on 1398 GB/s HBM = **<0.2 μs**. GEMM takes ~50-200 μs. Overhead <0.1%. |

So `.clone()` is essentially free. However, I'm not certain it's the right fix — let me explain the actual root cause I suspect:

## What I Think Is Happening

The crash is at `hidden_states *= self.residual_scale` in the **attention** layer, but the actual illegal access happens earlier in the **FP8 GEMM** and is reported late (async CUDA error). The GEMM code does:

```python
output.to(dtype=input_2d.dtype).view(*output_shape)
```

If `output` is already the correct dtype, `.to()` returns `self` (same tensor), and `.view()` creates a **non-contiguous view**. During CUDA graph capture, in-place operations on non-contiguous views can cause illegal memory access.

The proper fix is `.contiguous()` rather than `.clone()` — same effect but slightly cheaper (no copy if already contiguous):

```python
return output.to(dtype=input_2d.dtype).view(*output_shape).contiguous()
```

## Proposal

Apply a 1-line change to gptq.py in the W4A8 path to ensure contiguous output, then retest. Do you want me to:

1. **Apply the fix now** and retest on fcloud?
2. **Wait until tomorrow** as you originally suggested?
