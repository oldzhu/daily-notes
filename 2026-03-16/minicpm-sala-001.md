I’m tracing the runtime path differences between fp8+gpu graph+force-dense-minicpm and fp8+gpu graph so I can explain why the “dense” profile can still benchmark faster in this codebase.

Reviewed 4 files and explored token pool selection
I’ve confirmed two big control-plane changes already: force-dense-minicpm swaps you off the MiniCPM-specific attention backend, and it disables all sparse bookkeeping in the scheduler/config path. Now I’m checking the hot-path code that runs only in the sparse backend.

Searched for regex and reviewed 3 files
I’ve confirmed the sparse backend is doing extra per-step work that the dense path never does: compressed-K generation, sparse top-k scoring, sparse block-table construction, and extra memory-pool metadata. I’m reading the batch classification logic next, because your --dense-as-sparse flag is probably making that overhead apply to essentially everything.

Searched for regex and reviewed multiple code sections
Yes, it can make sense in this codebase. The key point is that your two combinations are not just “dense attention” versus “sparse attention”. They are two materially different runtime pipelines.

The biggest difference is that --force-dense-minicpm does two things at once. It disables MiniCPM sparse attention at the model-config level in model_config.py:236, and it also remaps the backend from minicpm_flashinfer to generic flashinfer in server_args.py:1518. So combination 1 is really “FP8 + GPU graph + generic dense FlashInfer path”, not “FP8 + MiniCPM sparse backend with sparse disabled but everything else unchanged”.

Why sparse can be slower here:

Your sparse profile is currently also using --dense-as-sparse from prepare_env.sh. In the MiniCPM backend, that sets dense_len = 0 in minicpm_backend.py:230. Then the sparse batch analyzer treats every batch as sparse in minicpm_sparse_utils.py:991. That means you are not letting the implementation keep short or moderate cases on a cheap dense fallback. You are forcing the sparse-routing pipeline onto everything.

The MiniCPM sparse path is not a single cheap sparse kernel. It is a multi-stage pipeline. It does sparse bookkeeping in the scheduler in schedule_batch.py, uses a heavier MiniCPM-specific request/token pool in model_runner_kv_cache_mixin.py, compresses K into K1/K2 scratch buffers in minicpm_sparse_utils.py, runs sparse top-k scoring in minicpm_backend.py:811, and builds sparse block tables in minicpm_backend.py:970. Dense FlashInfer avoids all of that routing overhead.

Under FP8, your sparse scorer is not even staying on a pure FP8 fast path. In sparse_get_topk_impl, the scorer inputs are cast to BF16 when KV cache dtype starts with FP8 in minicpm_backend.py:811. That was the right correctness/compatibility fix, but it adds extra dtype conversions and scratch traffic. So the sparse pipeline is paying both routing overhead and mixed-precision bridge overhead.

The sparse path may not be pruning as aggressively as the intuition suggests. The backend defines sparse_topk as topk + local_blocks, not just raw top-k, in minicpm_backend.py. So even “sparse” still keeps local window blocks plus selected blocks. If the effective retained set is still large, the compute savings may be smaller than the extra preprocessing cost.

So the current comparison is really this:

Combination 1: generic FlashInfer dense path, fewer special kernels, fewer metadata writes, fewer conversions
Combination 2: MiniCPM sparse backend, forced sparse on every batch because of --dense-as-sparse, extra K compression, extra top-k scorer, extra sparse page-table construction, extra BF16 bridge under FP8
In that situation, combination 1 being faster is not contradictory. It actually matches the implementation.

The clean conclusion is: your current “sparse” profile is not yet an optimized sparse fast path. It is a more complicated path with extra overhead, and right now that overhead is larger than the savings on your benchmark set.

The fairest future comparison, after the crash is fixed, would be:

fp8 + gpu graph + minicpm sparse without --dense-as-sparse
fp8 + gpu graph + force-dense-minicpm
That would tell you whether MiniCPM sparse really helps when it is allowed to activate only where the model/runtime thinks it should, instead of being forced on universally.

