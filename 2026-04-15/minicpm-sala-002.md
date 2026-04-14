Good questions. Let me research the specifics of torch.compile in SGLang. 

Research torch.compile in SGLang

Searched for files matching `**/server_args.py`, 4 matches

Searched for regex `torch.compile|torch_compile|TORCH_COMPILE`, 50 results

Searched for regex `graph_break|graph.break`, 4 results

Searched codebase for "how does torch.compile work integration with model runner executor scheduler", 24 results

Searched for regex `enable_torch_compile|torch_compile_max_bs|ENABLE_TORCH_COMPILE` (`**/python/sglang/srt/**`), 72 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1-1), lines 1 to 150

Searched for text `patch_torch_compile|monkey_patch_torch_compile`, 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#550-550), lines 550 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#4140-4140), lines 4140 to 4200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/utils/patch_torch.py#100-100), lines 100 to 140

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#173-173), lines 173 to 320

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#200-200), lines 200 to 250

Searched for regex `compile|dynamic|shape|graph_break|einops|reshape|view` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 30 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/utils/patch_torch.py#1-1), lines 1 to 60

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#1-1), lines 1 to 100

Searched for text `_to_torch _is_torch`, no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#1500-1500), lines 1500 to 1800

Searched for text `_to_torch` (`**/python/sglang/srt/model_executor/cuda_graph_runner.py`), 8 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#100-100), lines 100 to 500

Searched for regex `torch_compile|compile` (`**/python/sglang/srt/layers/attention/flashinfer_backend.py`), 8 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py#40-40), lines 40 to 100

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/torch_compile_research.md)

Compacted conversation

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_vrtx_016w1eE7QNGGa838iLmRN54L__vscode-1776120181357/content.txt#1-1), lines 1 to 200

Here's the full picture on `torch.compile` in SGLang:

## No Code Changes Needed

It's purely a server arg: add `--enable-torch-compile` to `SGLANG_SERVER_ARGS` in prepare_env.sh. Optionally add `--torch-compile-max-bs N` (default 32) to control which batch sizes get compiled.

## How It Works

- During server startup, for each batch size ≤ `torch-compile-max-bs`, SGLang wraps `model.forward` with `torch.compile(mode="max-autotune-no-cudagraphs")` and captures a compiled kernel  
- Batch sizes **above** the threshold still use regular CUDA graphs  
- So it's a **selective optimization for small batches** — the most relevant path for S1 (BS=1, 40% weight)

## Why It Speeds Things Up

The Inductor backend applies:
1. **Operator fusion** — merges sequential ops (LayerNorm + residual → single kernel)
2. **Reduced kernel launches** — fewer GPU dispatch calls
3. **Better memory reuse** — reduces temporary allocations
4. **Auto-tuning** — `coordinate_descent_tuning` tests multiple Triton kernel variants and picks the fastest

Our Test 12-VarC showed **5-7% speedup** across all tiers.

## Compilation Takes Time — At Startup

Yes, compilation adds to server startup time. Each batch size in the compile list (e.g., [1, 2, 4, 8, 16, 32]) gets compiled **once** during `CudaGraphRunner.__init__()`. This can add **minutes** to startup (typically 3-10 min depending on model size and number of batch sizes). Once compiled, the kernels are cached and reused — no per-request overhead.

SGLang enables `fx_graph_cache = True` which can cache compiled graphs across restarts (experimental).

## The OOM Problem We Hit

With `--torch-compile-max-bs 32`, the compiled kernels for all batch sizes 1–32 consume significant GPU memory. During accuracy eval (which sends varied batch sizes), the combined pressure of compiled kernels + KV cache + eval allocations exceeded VRAM → OOM crash.

**Fix**: Reduce to `--torch-compile-max-bs 8` or `16`. Fewer compiled batch sizes = less memory pressure, while still optimizing the most important small-batch path (S1 is BS=1, worth 40%).

## Quick Reference

| Aspect | Detail |
|--------|--------|
| Code changes? | **None** — server args only |
| Key args | `--enable-torch-compile --torch-compile-max-bs N` |
| Speedup | ~5-7% (observed) |
| Startup cost | +3-10 min compilation |
| Runtime cost | None (compiled kernels cached) |
| Risk | OOM if max-bs too high |
| Accuracy impact | None (same math, just fused kernels) |

When fcloud is back, the plan would be to test `--torch-compile-max-bs 8` first (safest for memory) then try `16` if that passes.
