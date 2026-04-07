

The GPU has a fixed amount of VRAM (e.g. ~48 GB on RTX PRO). The server allocates `--mem-fraction-static 0.84` of that for the KV cache pool.

**Available VRAM for KV pool ≈ 48 GB × 0.84 ≈ 40 GB**

Each KV slot stores K and V for all attention layers. Per slot:
- **BF16 (2 bytes/element):** `num_layers × (head_num_k + head_num_v) × head_dim × 2 bytes`
- **FP8 (1 byte/element):** `num_layers × (head_num_k + head_num_v) × head_dim × 1 byte`

Since each FP8 element is **half the size** of BF16, the same 40 GB fits **twice as many slots**:

| KV dtype | Bytes per element | Slots fitting in 40 GB | Max slot ID |
|---|---|---|---|
| BF16 | 2 | ~1.05M | ~1,050,000 |
| FP8 | 1 | ~2.10M | ~2,100,000 |

The int32 overflow threshold is **2,096,128** (= 2^31 / (8 heads × 128 dim)).

- **BF16:** max slot ~1.05M → well under 2.1M → **safe**
- **FP8:** max slot ~2.10M → **crosses 2,096,128** → overflow → crash

That's why this bug was invisible before FP8 was enabled — BF16's larger per-element size naturally kept slot IDs in the safe range. FP8 doubles the pool capacity, pushing slot indices past the int32 multiplication overflow point.

Shall I proceed with the fix?
