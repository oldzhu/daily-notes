`torch.compile` with `--torch-compile-max-bs N` pre-compiles CUDA graphs for batch sizes 1 through N. Each compiled batch size captures a separate CUDA graph that **pins all intermediate activation tensors in GPU memory permanently**. Here's why max-bs=32 causes OOM:

**Memory breakdown on our GPU (80GB A800):**
- Model weights (GPTQ 4-bit): ~5-6 GB
- KV cache (`mem-fraction-static=0.84`): ~67 GB allocated
- Remaining for CUDA graphs + overhead: ~7-8 GB

**CUDA graph memory cost per batch size:**
- Each graph captures all intermediate tensors (activations, attention scores, etc.) for that batch size
- For MiniCPM-SALA (32 layers), a single bs=32 graph might need ~2-4 GB of pinned activation memory
- With max-bs=32, PyTorch creates graphs for bs=1,2,...,32 — but only stores a subset (power-of-2 or padded sizes). Still, the largest graphs (bs=16, 32) each need multi-GB pinned memory

**Why accuracy eval triggers OOM but speed benchmarks don't:**
- Speed benchmarks use fixed, moderate concurrency (S1=1, S8=8, Smax=unlimited but short-lived)
- Accuracy eval uses `--concurrency 32`, creating sustained large batches that hit the max captured batch size repeatedly
- When a batch exceeds max-bs, sglang falls back to eager mode which needs dynamic memory — but the pinned CUDA graph memory is already consuming the headroom

**max-bs=8 works because:**
- Fewer/smaller graphs pinned → less permanent memory overhead
- Batches >8 fall back to eager mode, which can reuse freed memory dynamically
- The tradeoff: batches 9-32 don't get the compile speedup (mainly affects Smax, which is why Smax improvement dropped from -5.7% to -1.5%)
