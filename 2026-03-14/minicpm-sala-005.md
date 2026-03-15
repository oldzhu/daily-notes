Before appove, I would like to understand more of side effects:

stop write compressed k1/k2 into shared fp8 kv cache.
compressed k1/k2 will have separate scratch buffers
I would like to understabd
where fp8 kv cache is - gpu shared or global memory or cpu memory
the same question to compressed k1/k2 scratch buffers
in what kind of degree the impact to memory and speed.
besides compressed k1/k2 write to pointer in bf16 way are there any other writes doing the same thing
=======

Your questions are the right ones. The short answer is: the side effect is real, but it is bounded enough that this still looks worth trying given the speed numbers you just got.

1. Where is the FP8 KV cache?

It is in GPU memory, specifically normal device global memory, not CPU memory.

You can see that in memory_pool.py:664, memory_pool.py:668, and memory_pool.py:857:

when KV dtype is FP8, SGLang stores it as torch.uint8
the buffers are allocated on device=self.device
later they are viewed back as FP8 via memory_pool.py:969
So conceptually:

physical storage: GPU global memory bytes
logical interpretation: FP8 KV cache
It is also shared/persistent in the sense that it is the main runtime KV cache pool used by attention across requests, not a tiny local scratch array.

2. Where would the compressed K1/K2 scratch buffers be?

Also GPU memory, again device global memory, not CPU memory.

Today the compressed K tensors are already materialized as PyTorch tensors in minicpm_sparse_utils.py:337. Those allocations use device=device, which in this path is the CUDA device.

There are two practical cases:

without CUDA graph: they are temporary per-call GPU tensors
with CUDA graph decode: they can become reused persistent GPU buffers inside MiniCPM decode graph metadata, as seen around minicpm_backend.py:726 and minicpm_backend.py:774
So they are not “CPU staging buffers”. They are GPU-side scratch / working buffers.

3. What is the likely impact on memory and speed?

Memory:

the main FP8 KV cache is still the big memory win
the scratch buffers are much smaller than the full KV cache
K1 and K2 are compressed views at stride 16 and stride 64, so their token count is roughly far below the full token count
that means even if we keep BF16 scratch for compressed K1/K2, memory cost should be a fraction of the full BF16 KV-cache rollback cost
In other words:

reverting KV cache from FP8 to BF16 would blow up the whole cache footprint
adding BF16 scratch only for compressed K1/K2 increases memory, but much less
Speed:

there will probably be some penalty
but it should land in the sparse compression/scoring subpath, not the full decode attention read path
your current benchmark already shows FP8 with CUDA graph is basically at the strong path you wanted:
s1 142.00
s8 31.21
smax 25.68
That is why I still think it is worth pursuing. The likely trade is:

keep most of the decode-side FP8 benefit
pay a narrower overhead in sparse compressed-K maintenance
I would expect:

memory impact: moderate but bounded
speed impact: small to medium, but very likely still better than giving up FP8 entirely
4. Are there other writes doing the same dangerous thing?

In this MiniCPM sparse path, the main suspicious ones are the compressed-K Triton kernels.

The problematic pattern is in minicpm_sparse_kernels.py:1:

compress_k_complete_kernel_new
compress_k_complete_kernel_new_padded
Inside those kernels, the code does:

load from key_cache_ptr
accumulate in float32
then tl.store(...) back to key_cache_ptr
That is the exact “treat shared KV cache as writable compressed-K destination” pattern I’m worried about. You can see the write in minicpm_sparse_kernels.py:217.

By contrast, the normal KV-cache write path in memory_pool.py:995 is much safer for FP8 because it explicitly:

divides by scale
converts to KV dtype
views as uint8 for storage when needed
So normal KV writes are FP8-aware. The compressed-K Triton kernel writes are not obviously FP8-aware.

Other writes I checked:

req_to_sparse_k1_token / req_to_sparse_k2_token writes are just int32 metadata table writes in GPU memory, not the same issue, see memory_pool.py:619 and memory_pool.py:633
write_sparse_k1 / write_sparse_k2 are metadata writes, not FP8 tensor data writes
the biggest same-pattern risk is those two compressed-K Triton kernels
So my current answer is:

yes, there is a real side effect
no, it is not on the same scale as losing FP8 KV cache entirely
the FP8 KV cache itself lives in GPU global memory
the proposed scratch buffers also live in GPU global memory
the main dangerous same-style write I see is the compressed-K Triton write-back into key_cache_ptr
the normal KV write path is already more careful and is not the same class of problem
Given your benchmark result, this still looks worth trying.

If you approve, the next patch should be very specifically:

stop using key_cache_ptr as the write destination for compressed K in FP8 mode
write compressed K only into BF16 scratch buffers
leave the real FP8 KV cache untouched
That is the narrowest fix that matches the crash pattern and preserves the performance upside you just observed.


