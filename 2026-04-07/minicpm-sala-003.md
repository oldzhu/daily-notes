

Practically negligible. Here's the analysis:

| Concern | Impact |
|---|---|
| **Register pressure** | +1-2 registers per kernel (scalar index, not vectorized). Occupancy unchanged in practice |
| **Multiply throughput** | int64 multiply is ~2x slower than int32 on NVIDIA GPUs, BUT these are **scalar index computations** done once per chunk — not in the inner vectorized `tl.load`/`tl.store` loop. The actual bottleneck (loading `head_dim` floats, accumulation) is untouched |
| **Memory traffic** | Zero — the source tensor (`req_to_token`) is still int32 in DRAM. The `.to(tl.int64)` just widens the scalar in registers after load |
| **Pointer arithmetic** | Actually cleaner — GPU pointers are 64-bit, so Triton had to sign-extend int32 offsets anyway |
| **Correctness** | No risk — int64 is a strict superset of int32 |

**Bottom line:** The hot path is the vectorized `tl.load` of `head_dim=128` floats per token. The int64 multiply is a single scalar instruction per chunk iteration, completely dominated by the memory-bound vector operations. You won't see any measurable throughput difference.
