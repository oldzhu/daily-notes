Compacted conversationGood questions — short answers:

**1. K=1 vs K≥2**

- **K=1**: 1 Medusa head predicts the token at position `t+1` (one step beyond what the base model already predicts at `t`). So per decode step you get up to **2 tokens** accepted (base + 1 draft). Max accept length = 2.
- **K=2**: 2 heads predict `t+1` and `t+2`. Max accept length = **3** tokens per step. K=3 → 4, etc.
- **My recommendation for R1b: still K=1.** Reason: R1 uses **zero-initialized** heads (`W1=0` → head output = `SiLU(0)+h = h`, then `lm_head(h)` = same argmax as base at position `t`, NOT position `t+1`). So **expected accept rate is ~0% regardless of K**. R1 is a correctness gate, not a speed gate. K≥2 just multiplies snapshot memory and tree-mask complexity without buying anything until R3 (trained heads). I suggest: **land K=1 in R1b, make K configurable, set K≥2 as default after R3 training works.**
- If you want K≥2 in R1b anyway, it's ~30 extra LOC (tree mask + retrieve_index become non-trivial). Acceptable but adds risk.

**2. `NgramVerifyInput` vs custom `MedusaInput.verify`**

Both produce the same kind of object: a **verify tree** (draft tokens + parent_index + retrieve_index + tree_mask) that gets fed to `target_worker.forward_batch_generation(is_verify=True)`. Differences:

| Aspect | `NgramVerifyInput` | Custom `MedusaInput.verify` |
|---|---|---|
| Source of drafts | n-gram cache lookup on past tokens | Medusa head logits (top-1 or top-k per head) |
| Tree shape | flexible, depth ≥ 1 | depth = K (one node per head) |
| Existing code path | already wired into scheduler, model_runner, cuda_graph_runner | would need its own wiring |
| Acceptance logic | `accept_threshold=1.0` (exact match) | same — exact match against base argmax |

For K=1, the verify tree is just a **single linear chain of length 1** — `NgramVerifyInput` handles this trivially. **Recommendation: reuse `NgramVerifyInput`** in R1b. We feed it Medusa-produced drafts instead of n-gram lookup drafts. Saves ~150 LOC, reuses tested scheduler/cuda-graph wiring, and the acceptance math is identical. Only `MedusaWorker.forward_batch_generation` differs — it generates drafts via heads then constructs an `NgramVerifyInput` for the verify pass.

**3. 288 MiB snapshot — what / how often / total**

- **What**: one snapshot = clone of all 24 GLA recurrent-state rows for the current batch. At bs=24, state_dim ≈ 512 KiB per (layer, row), so 24 layers × 24 rows × 512 KiB = **288 MiB**.
- **How often per decode step**: **exactly 1 snapshot per step** is alive at any time. The flow is:
  1. snapshot before verify forward
  2. verify runs, GLA state advances by `(K+1)` tokens per row
  3. compute accept_len per row
  4. if all rows accept all `K+1` tokens → discard snapshot (state is correct)
  5. else → restore snapshot, then replay only the accepted prefix
  6. snapshot freed before next step
- **Total additional memory cost: ~288 MiB constant**, NOT 288 MiB × N. The snapshot is overwritten/freed every step. Steady-state extra GPU memory is one snapshot = 288 MiB.
- Comparison: 84 GB GDDR7, current KV cache + weights use the bulk. 288 MiB is ~0.3% — negligible.

**4. Eager-only vs CUDA graphs**

- **Eager**: every forward call goes through PyTorch dispatch + cuBLAS/triton kernel launches dynamically. Higher per-step launch overhead (~hundreds of µs per layer on small batches). Easy to implement: just skip `CUDAGraphRunner` for the verify path.
- **CUDA graphs (current baseline)**: the entire forward pass is captured once and replayed as a single GPU graph. ~30–50% faster at small batches (S1) because launch overhead is amortized to ~zero. Required for our S1 score to be competitive.
- **Why this matters for Medusa**: the verify path has a **different shape** (input length = `K+1` per row, not 1). CUDA graphs need a **separate capture for verify**. R1c will handle this.
- **Eager-only for R1b**: verify path runs in eager mode → S1 likely regresses ~20–40%. But R1 is a correctness gate (we expect S1 regression anyway from heads compute + zero accept rate). Acceptable for R1b. R1c restores graph capture.
- If we insist on CUDA graphs in R1b: ~200 extra LOC in `cuda_graph_runner.py` for verify-shape capture, plus risk of capture failures. **Recommendation: eager-only in R1b**, defer graph capture to R1c.

**5. Stage 2 as intermediate smoke-test commit?**

**Strongly prefer yes.** Reasons:

- Stage 2 (heads-shadow) is the **lowest-risk way to validate fcloud wiring**: server boots with `SOAR_SPEC_MEDUSA=1`, heads get instantiated, drafts are computed-then-discarded, output is **byte-identical to baseline** (because we don't actually verify or rewind anything yet).
- Tests we get from Stage 2:
  - ✅ Does the server start with the medusa flag?
  - ✅ Do the heads instantiate on GPU without OOM?
  - ✅ Does `capture_hidden_mode=LAST` plumbing work?
  - ✅ Does accuracy stay at baseline (must be **identical**, not just close)?
  - ✅ How much S1 regression from just running heads (no verify)?
- If Stage 2 passes byte-identical accuracy → Stage 3 only needs to debug the verify/rewind logic, which is the actually-tricky part.
- If we skip Stage 2 and Stage 3 fails on fcloud, we won't know whether the bug is in heads-instantiation, hidden-state plumbing, verify tree, or rewind. Splitting halves the debug surface.
- Cost: Stage 2 is ~80 LOC vs Stage 3's ~400 LOC. Cheap to land.

**My preferred plan if you approve:**

1. **Next session**: implement Stage 2 (K=1, heads-shadow, eager-only), commit as CHANGE_0155, fcloud test for byte-identical accuracy + S1 regression measurement.
2. **Session after**: implement Stage 3 (real verify + rewind, reusing `NgramVerifyInput`), commit as CHANGE_0156, fcloud test.
3. **Session after that**: R1c (CUDA graph capture for verify path) if Stage 3 numbers warrant it.

Want me to proceed with Stage 2 on these defaults (K=1, reuse `NgramVerifyInput`, eager-only, intermediate commit), or do you want to override any? 


