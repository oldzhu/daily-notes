...
Recommendation: stay on M2.0 (#2) since it's already drafted, then go to #1 (W4A8 FP8) and #3 (lightning state FP8) as the next two parallel work streams. Those two combined could realistically bring us to mid-tier of the leaderboard. #6 (speculative decoding) is the only path to top-5 without major architectural changes.

Want me to draft a formal proposal for any of #1, #3, or #4?
...
I am not quite understand, I think we already done at w4a8, right? we used gptq to do 4-bit offline quantization, and enabled marlin kernel and fp8 kv-cache in command line, so what do u mean of this? for 2(mixed fp8/fp4 kv), we know we have't tr it,  for 3(lighting state fp8), if we already using gptq to do 4-bit offline quantization, and enabled marlin kernel and fp8 kv-cache in command line, the lighting state will still be bf16 and not fp8? In addition, for using gptq to do 4 bit quantization, it is not nvfp 4 and it is int 4,am I right?
========
You're catching real ambiguities. Let me clarify each precisely.

## 1. "We already do W4A8 because GPTQ + Marlin + FP8 KV?" — **NO.**

This is the most common misconception so let me draw the data flow:

```
Input activation (BF16)  ──┐
                           │
                           ▼
                   ┌───────────────────────┐
                   │ Marlin GEMM kernel    │
GPTQ INT4 weight ─►│  Inside kernel:       │── BF16 output activation
                   │  1. Load INT4 weight  │
                   │  2. Dequant to BF16   │
                   │  3. Compute BF16 × BF16│
                   └───────────────────────┘
```

**Marlin's compute math is BF16 × BF16. The "weight is 4-bit" only describes storage/bandwidth, not compute precision.** Tensor cores run BF16 fmas at **148 TFLOPS** on SM120.

If we used a true **W4A8** kernel (e.g., Marlin-W4A8 / QQQ / mxfp8 path), the inner math would be BF16 × FP8 or INT8 × INT4 → using the **FP8 tensor cores at 296 TFLOPS** — literally 2× the throughput of what we run today. That's the unexplored gain.

### What about "FP8 KV cache" — doesn't that count as FP8 activation?

**No.** They're completely separate code paths:

| Path | What it touches | Dtype | Code |
|---|---|---|---|
| Marlin GEMM | qkv_proj, o_proj, MLP weights × activations | INT4 W × **BF16** A | `sgl-kernel/.../gptq_marlin.cu` |
| Paged KV pool | K/V tensors **after** QKV proj, **before** attention | FP8 e5m2 (storage) → cast to **BF16** when read | memory_pool.py |
| Attention math | softmax(QKᵀ)V | **BF16** | FlashInfer / FA backend |

`--kv-cache-dtype fp8_e5m2` only changes how K/V are **stored in memory between layers**. The moment they're read back for attention compute, they're cast to BF16. It saves DRAM bandwidth and memory footprint, not GEMM TFLOPS.

So today our compute precision is **uniformly BF16** everywhere except weight storage. The "8" you're thinking of (FP8 KV) is bandwidth-only.

---

## 2. "GPTQ INT4 ≠ NVFP4 — am I right?" — **Yes, fully correct.**

| Format | Encoding | Hardware | Used for |
|---|---|---|---|
| **GPTQ INT4** | Uniform integer, `value = scale × (q − zero_point)`, group_size=128 | Marlin custom dequant + BF16 tensor core | Our weights today |
| **NVFP4 (E2M1)** | Floating-point: 1 sign + 2 exp + 1 mantissa, with shared block-scale | **QMMA tensor core directly on FP4** at 593 TFLOPS on SM120 | Test 21 (failed catastrophically) |

Two completely different number systems and two completely different hardware paths. A "W4 model" is not a single thing.

---

## 3. "Does FP8 KV flag also push lightning state to FP8?" — **NO.**

The lightning recurrent state is allocated by a **separate** memory pool, not the standard paged KV pool. `--kv-cache-dtype fp8_e5m2` only controls `MHATokenToKVPool` (the standard attention's K/V cache).

Lightning state path:
- Allocator: memory_pool.py → `MambaPool` / linear-attn state pool (separate class).
- Allocated in **BF16** (or FP32 for accumulation), regardless of the `--kv-cache-dtype` flag.
- Read/written every decode step inside `fused_recurrent_simple_gla` Triton kernel.

So **today's data layout for the 24 lightning layers**:
```
Lightning state S ∈ R^(num_kv_heads × d × d), dtype = BF16
```
24 layers × bs × ~64KB/layer = lots of BF16 state churning through L2 every decode step.

That's why "**lightning state FP8**" (proposal #3) is a real, separate optimization that does NOT come for free with `--kv-cache-dtype fp8_e5m2`. We'd have to:
1. Modify the state allocator to allocate FP8 buffers.
2. Modify `fused_recurrent_simple_gla` Triton kernel to dequant on read and quant on write.
3. Add per-head scale management.

---

## Corrected summary table

What we ACTUALLY ship today:

| Component | Storage dtype | Compute dtype | Hardware unit |
|---|---|---|---|
| Most weights (MLP, std-attn QKV non-sparse, lightning QKV/O) | INT4 (GPTQ) | BF16 (post-dequant) | BF16 tensor core 148 TF |
| 8 std-attn QKV weights (sparse_qkv_w8) | INT8 (GPTQ) | BF16 (post-dequant) | BF16 tensor core |
| All activations | BF16 | BF16 | BF16 tensor core |
| Std-attn KV cache | **FP8 e5m2** (storage only) | BF16 (cast on read) | BF16 attention math |
| Lightning recurrent state | **BF16** | BF16 | BF16 in Triton kernel |
| Attention math (8 layers) | BF16 | BF16 | FlashInfer BF16 |
| Lightning math (24 layers) | BF16 | BF16 | Triton BF16 |

What "real W4A8 + FP4 KV + FP8 lightning state" would look like:

| Component | Storage dtype | Compute dtype | Hardware unit |
|---|---|---|---|
| Most weights | INT4 (GPTQ) | **FP8** (post-dequant) | **FP8 QMMA 296 TF** |
| Activations | **FP8 e4m3** | FP8 | FP8 QMMA |
| Std-attn KV cache (mixed) | FP8 / **FP4 E2M1** for inner layers | BF16 cast | BF16 attention |
| Lightning recurrent state | **FP8** | FP8 (or BF16 accum) | FP8 in Triton |

That's 3 separately-engineered changes, none of them automatic from current flags.

---

## Bottom line

- **#1 (W4A8 FP8 act)** is real and unstarted — it would actually exercise SM120's 296 TFLOPS FP8 tensor cores instead of the 148 TFLOPS BF16 ones we use today.
- **#2 (mixed FP8/FP4 KV)** affects only the 8 std-attn layers, also unstarted (M2.0 proposal).
- **#3 (lightning state FP8)** is not implied by any current flag — lightning state is BF16 today and stays BF16 unless we modify the kernel.
- All current "4-bit" in our submission is **INT4 weight-only**, never FP4 and never with FP8/INT8 activations.

Want me to amend the research note to make this distinction explicit (BF16 compute today vs. FP8/FP4 compute proposed) so it's clearer on offline re-read?   


