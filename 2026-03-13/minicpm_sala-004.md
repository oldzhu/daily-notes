I found the MiniCPM-specific cache write and read sites. That’s enough to answer how KV dtype affects each phase and why a prefill-only wrapper failure can still block the whole FP8 path.No, kv-cache-dtype is not “only in prefill”. It affects the KV cache as a stored data structure, so it impacts both phases, but in different ways.

In this MiniCPM path, prefill and decode use different FlashInfer wrappers in minicpm_attention_kernels.py:

Decode uses BatchDecodeWithPagedKVCacheWrapper
Prefill uses BatchPrefillWithPagedKVCacheWrapper
Both wrappers are initialized with kv_data_type=self.data_type, and self.data_type is taken from model_runner.kv_cache_dtype
So the chosen KV dtype is part of both wrapper plans, not just prefill.

What changes between prefill and decode is the way that dtype matters:

Prefill / extend phase:
The new K/V tensors produced by the model are written into the KV cache via set_kv_buffer(...) in the MiniCPM backend, for example in minicpm_backend.py.
Impact:

Memory footprint of the stored cache changes immediately.
Write bandwidth into cache changes.
There may be quantization or cast overhead when storing into FP8 instead of BF16/FP16.
In long-context workloads, this can reduce cache growth pressure a lot.
Decode phase:
Attention reads the already stored K/V cache back through the decode wrapper and uses it in attention computation. In MiniCPM FlashInfer this is the wrapper.forward(...) decode branch in minicpm_attention_kernels.py.
Impact:

This is usually where FP8 KV cache helps most.
Decode is dominated by repeatedly reading large cached K/V states for every next token.
If you halve bytes per cached element, you reduce memory traffic and pressure on bandwidth.
That can improve throughput and sometimes let larger batches or longer contexts fit.
So the practical split is:

Prefill benefit:
Usually smaller and more mixed.
Reason:

Prefill often spends more time on GEMMs and prompt-side attention computation than on repeatedly rereading old KV.
You do save memory and cache-write bandwidth, but prefill is less purely KV-bandwidth-bound.
Decode benefit:
Usually larger.
Reason:

Decode repeatedly streams KV cache from memory.
Smaller KV entries reduce memory bandwidth demand.
This is why toolkit guidance frames FP8 KV cache as especially helpful in long-context scenarios.
In rough system terms, the main resource effects are:

GPU memory capacity:
Big effect.
FP8 KV cache is about half the storage of BF16/FP16 KV cache.
That can increase max context, concurrency headroom, or reduce OOM risk.

Memory bandwidth:
Big effect in decode, smaller in prefill.
Decode keeps rereading old K/V every token, so bandwidth savings matter directly.

Compute:
Usually modest effect by itself.
KV cache dtype is not primarily a compute optimization like W4 Marlin on linear layers.
It mainly reduces data movement, not GEMM FLOPs.

Accuracy:
Nonzero risk.
Because cached K/V activations are quantized more aggressively than BF16/FP16.

Now to your specific crash: why does it look “prefill-only”?
Because the current MiniCPM FlashInfer path creates the prefill wrapper with backend="fa2" in minicpm_attention_kernels.py, and that backend rejects the FP8 tensor-core path. So the failure is triggered when planning or running prefill, even though the configured KV dtype is conceptually meant for the whole cache lifecycle.

So the right mental model is:

kv-cache-dtype is a cache-format choice for both write and read
The speed gain usually shows up more in decode than prefill
Your current failure happens in prefill because MiniCPM’s prefill wrapper backend is the incompatible part, not because decode is the only phase that uses KV dtype
If you want a concise expectation for benefit magnitude:

Memory usage: potentially close to 2x KV capacity improvement versus BF16
Decode speed: can be meaningful when workload is long-context and bandwidth-bound
Prefill speed: often smaller, sometimes negligible, sometimes slightly worse if cast/quant overhead dominates
If you want, next I can give you a very concrete “what exactly to benchmark” matrix for FP8 KV cache:

memory impact
prefill-only impact
decode-only impact
end-to-end S1/S8/Smax impact

