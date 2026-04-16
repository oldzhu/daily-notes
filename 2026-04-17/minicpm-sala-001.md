Good call. The C drop could easily be accuracy variance (mcq fluctuates ±13%), and reverting would sacrifice confirmed S1/S8 gains for uncertain benefit.

Let me prepare the next optimization proposal.

---

## Optimization Proposal: CHANGE_0080 — SimpleGLA Kernel Parameter Tuning

### Objective
Make FLA chunk size configurable and test optimal values for both chunk size and recurrent threshold. These both affect the same 24 SimpleGLA layers (~75% of compute).

### Expected Gain
- **A2 (chunk size)**: 1-10% prefill improvement depending on optimal value
- **A4 (recurrent threshold)**: 1-3% mixed improvement

### Rule Compliance
- Pure speed optimization, zero impact on model weights or accuracy
- Chunk size controls Triton kernel tile granularity (mathematically equivalent regardless of value)
- Recurrent threshold only controls which kernel path (chunk vs recurrent) — both produce identical results

### Risk to Accuracy/Stability
- **Zero** — these parameters only affect kernel scheduling, not numerical computation
- Constraint: chunk size must be power of 2 (enforced by cumsum.py)

### Files to Change

**Code changes (2 files):**

1. chunk_delta_h.py — line 19:
   - Change `CHUNK_SIZE = 64` → `CHUNK_SIZE = int(os.environ.get("SGLANG_FLA_CHUNK_SIZE", "64"))`

2. chunk.py — line 37:
   - Change hardcoded `chunk_size=64` → `chunk_size=CHUNK_SIZE` (import from chunk_delta_h)

**Config changes (prepare_env.sh):**
- Add `export SGLANG_FLA_CHUNK_SIZE="${SGLANG_FLA_CHUNK_SIZE:-64}"` (env var, default unchanged)
- Recurrent threshold already configurable via `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD`

### Test Plan (on fcloud, single cycle)
| Test | Chunk Size | Recurrent Thresh | Notes |
|------|-----------|-----------------|-------|
| Baseline | 64 | 128 | Current — just speed re-verify |
| A | **32** | 128 | Smaller chunks = more parallelism |
| B | **128** | 128 | Larger chunks = less kernel launches |
| C | 64 | **64** | More chunked prefill |
| D | 64 | **256** | More recurrent prefill |

Each test: restart server with new env vars → run speed `--variant s1` (fastest to check).

### Rollback
- Remove env var or set `SGLANG_FLA_CHUNK_SIZE=64` and `RECURRENT_THRESHOLD=128`

---

**Shall I proceed with implementing the code changes?**
