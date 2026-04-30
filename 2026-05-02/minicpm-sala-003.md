Good analysis. Let me respond to each of your three concerns directly.

---

## (1) Local vs official S8/Smax inversion — what it tells us

You've spotted a real signal. Let me lay out the data:

| Tier | fcloud v18 | fcloud pre-v18 | local Δ | official v18 | official pre-v18 | official Δ |
|------|------------|----------------|---------|--------------|------------------|------------|
| S1 | (better) | — | v18 wins | 586.64s | 596.25s | **v18 wins** ✓ |
| S8 | (better) | — | v18 wins | 1087.90s | 1066.35s | **pre-v18 wins** ✗ |
| Smax | (better) | — | v18 wins | 2857.64s | **2746.62s** | **pre-v18 wins** ✗ |

**Interpretation**: v18's "improvements" over pre-v18 are tuned for **short context** (our local speed_s8/smax has shorter inputs than official). On official's long-context S8 (1066s) and Smax (2746s), pre-v18 is genuinely faster.

**What changed pre-v18 → v18** (from memory of recent iterations): aggressive scheduler args — `--prefill-max-req=1`, `--sched-cons=1.0`, `--enable-fused-qk-norm-rope`, torch.compile max-bs=8, chunk=32K. These help short-batch S1 but fragment long-context prefill.

**Implication for "v18 as baseline"**:
- v18's source-code changes (fused qk-norm-rope, torch.compile) are net positive — keep them
- v18's `prepare_env.sh` SGLANG_SERVER_ARGS may be **over-tuned for our local benchmark** — needs revisiting before submission
- Specifically: `--prefill-max-req=1` and `--sched-cons=1.0` likely hurt official long-context. Pre-v18 likely had `prefill-max-req=4, sched-cons=0.8` which our 2026-04-20 config sweep called best.

**Recommendation**: Use **v18 source code** as baseline, but **plan to A/B test the SGLANG_SERVER_ARGS** between pre-v18 values and v18 values on a long-context local set before our next official submission. This is essentially M3 (build a long-context-faithful local set so we stop optimizing for the wrong target).

---

## (2) Reverting repo to v18 commit?

**No — don't `git reset` to the v18 commit.** Better approach:

The current branch `mixed_minicpm_cudagraph` already contains v18 code (commit corresponds to the v18 tarball that produced score 39.01). The post-v18 work that produced v19 was added on top, and v19 was shown by our 3-way isolation tests to be **strictly worse** (FLA changes neutral, minicpm.py changes caused hangs and acc regression).

**Cleanest options** (preserve git history):

**Option A — explicit revert commits** (recommended):
```bash
git log --oneline <v18_tag>..HEAD             # list v19 commits to revert
git revert <v19_commits_in_reverse_order>      # creates revert commits
git push minicpm-src mixed_minicpm_cudagraph
```
History preserved; `git diff <v18>..HEAD` shows clean state; if any v19 piece is salvageable later, the original commit is still there.

**Option B — branch from v18, work forward**:
```bash
git checkout -b v18_baseline <v18_commit>
git checkout -b m2_kv_mixed_precision v18_baseline
```
Cleanest for new optimization track; requires deciding what to do with `mixed_minicpm_cudagraph`.

**Option C — keep current branch, partial revert by file**:
```bash
git checkout <v18_commit> -- python/sglang/srt/models/minicpm.py \
                              python/sglang/srt/models/minicpm_eagle3.py \
                              python/sglang/srt/speculative/eagle_worker.py
git commit -m "Revert v19 minicpm/eagle changes; baseline back to v18"
```
Surgical — only the v19 source-code changes proven harmful. Keeps any v19 doc/script changes that were neutral.

**My recommendation: Option C.** Surgical, preserves anything in v19 that was neutral (e.g., the FLA changes that our test showed were beneficial when paired with v18's other code). Need to identify the exact v19 commits first, then I can list which files to revert.

**Action item before any new optimization**: I need to:
1. Identify the v18 commit hash (from when v18 tarball was built)
2. Diff v18→HEAD and list each changed file with a hypothesis (revert / keep / TBD)
3. Get your approval on the file list, then execute Option C

---

## (3) Mixed-precision KV cache as next iteration — agreed, with sequencing

This is **the right next iteration**. The Week 5 champion's results map cleanly to our problem:
- Pure NVFP4 KV → ~75% acc → fails our C ≥ 0.92 floor → ❌
- FP8 KV → ~80% acc → matches our v18 baseline ✓
- **Mixed (FP8 first/last layers + NVFP4 middle) → ~80% acc with FP4 bandwidth** ✓

For our 32-layer architecture (8 std attention + 24 lightning SimpleGLA), the layer-wise sensitivity ablation is doable in ~1 day per pass.

**Why this is the highest-EV move**:
- Smax is our worst tier officially (2857s); the new speed dataset has 25% inputs in 32K-128K and 26% in 128K-256K → KV bandwidth dominates
- KV cache reads halve → directly attacks Smax memory-boundedness
- Bandwidth-only optimization, doesn't touch Marlin GEMM at all
- Champion has validated the recipe for the same model family

**Proposed iteration M2 — Mixed FP8/NVFP4 KV cache**:

**Scope (4 sub-stages, each individually shippable)**:

**M2.0 (1-2 days) — Build sensitivity ablation harness**
- Script that forces `kv_dtype = FP4` for one layer at a time, FP8 for rest
- Run accuracy eval on public set after each layer flip
- Output: per-layer accuracy delta CSV → identifies "sensitive" set
- No SGLang code changes; pure config knob

**M2.1 (3-5 days) — NVFP4 KV memory pool**
- New file: `python/sglang/srt/mem_cache/nvfp4_kv_pool.py`
- FP4 E2M1 packed 2-per-byte; per-block scale factor (e.g., block_size=16); BF16 scales array
- Allocate/release/index parity with existing FP8 pool
- Unit tests: round-trip quant→dequant accuracy; pool reuse correctness

**M2.2 (3-5 days) — Attention backend mixed-dtype path**
- File: `python/sglang/srt/layers/attention/minicpm_flashinfer.py`
- Per-layer dtype selection from config dict
- Quantization on KV write; dequant-on-load in attention kernel
- Integration with FlashInfer / SGLang attention dispatch
- May need a small CUDA kernel for fused dequant-load if existing FlashInfer path doesn't accept FP4 directly

**M2.3 (2-3 days) — Per-layer config + validation**
- `prepare_env.sh`: new env `SGLANG_KV_LAYER_DTYPE_JSON='{"fp8": [0,1,30,31], "fp4": "rest"}'` (example)
- Full accuracy + speed test on fcloud
- Tune the FP8/FP4 layer split based on M2.0 ablation results

**Total**: ~2 weeks. Expected gain: **+15-30% Smax**, +5-10% S8, neutral-to-small S1.

**Risk mitigation**:
- All paths preserve FP8 fallback via env var
- M2.0 can run independently to gather data before any code change
- Each sub-stage is shippable: even M2.1 alone (without M2.2) is dead code that compiles, no risk to baseline

---

## Recommended sequence

| Day | Task | Output |
|-----|------|--------|
| 1 | (2) Identify v18 commit; produce v18→HEAD diff with revert/keep classification | File list for your approval |
| 1 | Execute Option C surgical revert; smoke-test acc on fcloud | Branch state = v18 baseline |
| 2 | Draft `PROPOSAL_iteration_M2_mixed_kv_cache.{en,zh}.md` | Awaiting approval |
| 3-4 | (M2.0) Build sensitivity ablation harness | Per-layer FP4 sensitivity CSV |
| 5-9 | (M2.1) NVFP4 KV pool implementation | Tested pool class |
| 10-14 | (M2.2 + M2.3) Attention adapter + per-layer config + final test | Submission v20 candidate |
| 15 | Local long-context speed set rebuild (M3) | Better local↔official correlation |

---

## Awaiting your decisions

1. **Confirm Option C** (surgical revert of v19-only changes) — or prefer Option A (revert commits) / B (new branch)?
2. **Confirm M2 scope** — proceed with full 4-stage plan, or start with M2.0 ablation only and re-decide?
3. **Should I now**: (a) identify v18 commit + produce diff classification, **and** (b) start drafting `PROPOSAL_iteration_M2_mixed_kv_cache.{en,zh}.md`?

Once you confirm, I'll execute (a) and (b) in parallel before any code change.  


