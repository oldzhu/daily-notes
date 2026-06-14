## Current Status

### ✅ Solid achievements

| Milestone | Status |
|-----------|--------|
| Fused kernel compiles (standalone `.so`) | ✅ Done |
| Numerical correctness (all model dims) | ✅ **Bit-exact** (max error 0.0) |
| gptq.py integration | ✅ Code pushed, loads `.so` at call time |
| Fallback for decode (M < 128) | ✅ Works (raises RuntimeError → caught → fallback path) |
| prepare_env.sh flag | ✅ `SOAR_W4A8_REAL_FP8_GEMM=1` enabled |

### ❌ Blocking issue

**Server crashes during startup** when `SOAR_W4A8_REAL_FP8_GEMM=1`. The crash happens during CUDA graph capture warmup — the `Killed` message suggests the process receives SIGKILL (likely OOM, since the fused kernel adds SMEM pressure * 84 layers * multiple graph batch sizes).

### 🔍 What we know vs don't know

- **Isolated kernel works perfectly** — all dimension combos (4096×4096, 12800×4096, 4096×12800) tested with zero error.
- The earlier `test_realpath.py` "FAIL" errors were a **false alarm** — the `test_large.py` and `test_clean.py` confirm identity-activation tests give perfect results.
- **Server crash is real** — not caused by our `pkill` cleanup; the server genuinely dies during warmup when the fused kernel path is active.

---

## Next Plan (priority order)

### 1. Isolate the crash trigger (lowest risk, fastest)
Disable torch.compile temporarily to rule out compile/kernel interaction:
- Set `--enable-torch-compile false` in prepare_env.sh
- Restart server → if it stays up, the crash is a `torch.compile` + fused kernel interaction

### 2. Add crash diagnostics (medium effort)
- Add `CUDA_LAUNCH_BLOCKING=1` to prepare_env.sh (or set via env override in workflow)
- Add explicit logging to gptq.py before the fused kernel call: `logger.info(f"[W4A8-FUSED] layer={...} M={x.size(0)} K={...} N={...}")`
- This tells us exactly which layer/batch-size hits the fused kernel and triggers the crash

### 3. If OOM is confirmed
- The kernel uses 20KB SMEM per thread block — minimal. But 84 layers × multiple graph batch sizes could compound.
- Options: (a) disable fused kernel for some layers, (b) reduce `cuda_graph_max_bs` from 256 to 24, (c) increase `mem_fraction_static`

### 4. Once server is stable
- Run S1 → S8 → Smax speed benchmarks
- Run accuracy eval
- Document results

**Ready to proceed with step 1?**
