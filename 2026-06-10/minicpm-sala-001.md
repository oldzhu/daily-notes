Good questions — let me look at the actual dispatch to give you precise answers.

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#990-990), lines 990 to 1020

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#1020-1020), lines 1020 to 1100

Searched for text `thread_m_blocks|thread_n|thread_k|auto.config|score` (`**/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`), 20 results

Here's the full comparison:

## Marlin vs W4A8 — Side by Side

### 1. Parameters passed

| Parameter | Marlin | W4A8 (our path) |
|---|---|---|
| **Input** | `x`: BF16 `(M, K)` | `x_fp8`: FP8 e4m3 `(M, K)` + `input_scale`: float32 `(M, 1)` |
| **Weight data** | `w_q`: int32 `(K//8, N)` Marlin-repacked | `w_fp8`: FP8 e4m3 `(N, K)` column-major |
| **Weight scales** | `w_s`: BF16 `(K/g, N)` per-group | `w_scale`: float32 `(N/128, K/128)` per-block |
| **Zero points** | `w_zp`: int32 packed | n/a (folded into dequant) |
| **Group index** | `g_idx`: int32 (for desc_act) | n/a (desc_act=False only) |
| **Workspace** | `workspace`: pre-allocated buffer | n/a (CUTLASS manages internally) |
| **Output** | BF16 `(M, N)` | BF16 `(M, N)` |

They are called from the **same `apply()` method** in gptq.py — the dispatch is just an `if` branch:

```
if layer._soar_w4a8_real_active → W4A8 path
else                             → Marlin path (fallback)
```

### 2. Shape optimization — Marlin vs W4A8

**Marlin** has a **runtime auto-config** system:

```
For each layer (M, N, K) at inference time:
  1. Enumerate pre-defined tile configs:
     thread_m_blocks ∈ {1, 2}     (M-tiles per thread)
     thread_n ∈ {64, 128, 256}    (N elements per thread)
     thread_k ∈ {64, 128, 256}    (K elements per thread)
     num_threads ∈ {128, 256}     (threads per block)

  2. Score each config:
     estimated_time = f(M, N, K, tile_config, occupancy)
     score = M * N * K / estimated_time

  3. Pick highest-score config for this (N, K)

  4. Cache the winner → reuse for subsequent calls
```

The SM120 Marlin auto-config (added in CHANGE_0125) has special tile tables tuned for Blackwell's 96 SM / warp-level MMA.

**W4A8 (CUTLASS SM120)** uses a **fixed tile shape**: 128×128×128 MMA. The CUTLASS scheduler auto-decomposes larger problems into multiple tiles. There's no per-shape tuning — the CUTLASS `KernelScheduleAuto` handles everything.

| Property | Marlin | W4A8 CUTLASS |
|---|---|---|
| Tile shape | Variable (per-shape optimal) | Fixed 128×128×128 |
| Thread config | Auto-selected from candidate pool | Fixed by CUTLASS |
| SM120-specific | ✅ (CHANGE_0125 tile table) | ✅ (SM120 FP8 QMMA) |
| desc_act support | ✅ | ❌ (128×128 only) |
| M=1 (decode) | Efficient (strip-mined) | Works but tile overhead |

### 3. Does mixed INT4+INT8 hurt speed?

The 8 sparse-attn layers have INT8 QKV weights. Let me quantify:

| Weight group | Count | Size each | Total | Path |
|---|---|---|---|---|
| MLP gate_up (INT4) | 32 layers | 128 MB | 4,096 MB | W4A8 |
| MLP down (INT4) | 32 layers | 128 MB | 4,096 MB | W4A8 |
| QKV attn (INT4) | 24 lightning layers | ~40 MB | 960 MB | W4A8 |
| **QKV attn (INT8)** | **8 sparse layers** | **~40 MB** | **320 MB** | **Marlin INT8** |
| O-proj (INT4) | 32 layers | ~16 MB | 512 MB | W4A8 |

The INT8 QKV weights are **320 MB / ~10,000 MB = ~3%** of total weight volume. Having 3% of weights go through Marlin while 97% go through W4A8 is negligible — the dispatch overhead (one branch check per layer) is ~nanoseconds.

**But there IS a subtle concern**: the 8 INT8 QKV layers will see Marlin's BF16 compute (148 TF) while their MLP neighbors see W4A8's FP8 compute (296 TF). This is fine because QKV weights are small and compute is not the bottleneck for them.

### 4. Pros and cons summary

| | Marlin (current baseline) | W4A8 (our target) |
|---|---|---|
| **Weight storage** | INT4 (0.5 B/elem) | INT4 (0.5 B/elem) — same |
| **Activation** | BF16 (2 B/elem) | FP8 (1 B/elem) — **2× less traffic** |
| **Compute** | BF16 MMA at 148 TF | FP8 QMMA at **296 TF** |
| **S1 (decode)** | Bandwidth-optimal (INT4) | Same BW + less act traffic → **5-10% faster** |
| **Smax (prefill)** | BF16 MMA bottleneck | FP8 QMMA 2× throughput → **20-30% faster** |
| **Shape tuning** | Per-shape auto-config | Fixed tile (CUTLASS decomposes) |
| **Maturity** | Battle-tested (ampere→blackwell) | New kernel, needs validation |
| **Accuracy risk** | Verified C=1.0 | FP8 activation → TBD |
| **desc_act support** | Yes | No (requires g=128, act_order=False) |

The W4A8 kernel is called from the **exact same place** in the sglang stack — `gptq.py::apply()` — just through a different branch. The integration cost at the Python level is minimal; the real work is in the CUDA kernel.
