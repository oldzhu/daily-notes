Ran terminal command:  cd /home/oldzhu/sglang && git status --short && git log --oneline -3 minicpm-src/mixed_minicpm_cudagraph 2>/dev/null | head -3

CHANGE_0140 regression doc is already on `minicpm-src` (commit `90e99240d`). The other modified/untracked files in soar_2026_changes (CHANGE_0130, OPTION_B_PHASE_1_DELIVERY, PROPOSAL_iteration_M20, PLAN_post_v18) are not from this session — leaving them alone unless you ask. Proceeding to (a). 

Searched for files matching `**/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`, 1 match

Ran terminal command:  wc -l /home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md /home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 254

Let me also peek at what's currently deployed and the most relevant referenced docs.         

Searched for regex `SGLANG_SERVER_ARGS|--enable|--max-|--prefill|--schedule|--chunk|--kv-cache|--quantization|--force-dense|--torch-compile|--mixed-chunk` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 20 results

Important: the **shipped v20 config** uses `prefill-max-requests=1`, `schedule-conservativeness=1.0`, `chunked-prefill-size=32768` — NOT the catalog's "best" of 4 / 0.8 / 65536. Likely those got rolled back somewhere. Let me check the leaderboard memory and a couple recent results.    

Read memory

Read memory [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

I've read the catalog, leaderboard memory, and current prepare_env.sh. Here's my ranked top-3.

---

## Context anchors

**Where v20 stands:**
- Official: S1=586, S8=1080, Smax=2840, acc=100 (C=1.0), **score 32.84** — regressed from v18's 39.01 because competitors got faster (S1 actually IMPROVED from prev-18's 596 → 586; the relative-ratio scoring punished us anyway)
- Top-5 needs roughly **score ≥75–80** in current leaderboard climate → ~**2× speedup**
- Dataset: **68% of inputs are 32K-512K** (prefill-dominant)
- Profiling: prefill GEMM = **85.3%**, FLA = 12.4%, FlashInfer attention = 8.8%; decode GEMM = 63.5%, torch.compile-fused = 29.4%

**v20 shipped config (`prepare_env.sh:241`):**
```
--prefill-max-requests 1   --schedule-conservativeness 1.0   --chunked-prefill-size 32768
--max-prefill-tokens 32768 --max-running-requests 24         --mem-fraction-static 0.84
--quantization gptq_marlin --force-dense-minicpm             --kv-cache-dtype fp8_e5m2
--enable-fused-qk-norm-rope --enable-torch-compile --torch-compile-max-bs 8 --enable-mixed-chunk
```

**Important discrepancy:** the catalog claims Tier 1 is exhausted with "best = `prefill-max-req=4, sched-cons=0.8, chunk=65536`" (-8.2% S1 on old data), but **v20 ships with 1 / 1.0 / 32K**. The "best" config got rolled back somewhere and there's no documented submission result for it on the new long-context dataset. This is an unresolved free lever.

---

## Top-3 ranked candidates (recommended order)

### #1 — Re-test catalog "best Tier 1 config" on the new long-context dataset (QUICK WIN)
- **Change**: `--prefill-max-requests 4`, `--schedule-conservativeness 0.8`, `--chunked-prefill-size 65536`, `--max-prefill-tokens 65536`
- **Why now**: zero code change, ~30 min on fcloud, and there's no recorded official measurement of this combo against the new long-context speed set. Old-data result was -8.2% S1; on prefill-dominated long-context the gain should be **larger** (bigger chunks amortize the per-step cost over longer prompts).
- **Expected gain**: 5–15% on S1/S8/Smax (highly leveraged given long inputs)
- **Risk**: low — possible Smax OOM if memory tight; we already use `mem-fraction-static 0.84`. Easy fallback.
- **Effort**: config edit + 1 fcloud round (acc + speed).
- **Probability of success**: HIGH
- **Why it was rolled back?**: I don't see a documented reason. Need to re-investigate. The catalog's 39.25 → 39.01 → 32.84 trajectory doesn't tie this back. Worth a clean test.

### #2 — NVFP4 KV cache (P2 in catalog, survey already done)
- **Change**: `--kv-cache-dtype fp4_e2m1` (env-gated via existing `CHANGE_0131` plumbing per prepare_env.sh line 152), dense-only first.
- **Why now**: ~**44% KV memory savings** vs FP8 → fits more concurrent long-context requests in cache → directly attacks Smax tier (where competitors widened the gap). Survey says ~80% of plumbing in tree; real gaps localized to 4 sparse-FP8 gates in minicpm_backend.py.
- **Expected gain**: 5–20% on Smax, smaller on S1/S8. KV bandwidth pressure also drops on long inputs.
- **Risk**: medium — accuracy threshold ≥75% locally (we have ~80% margin currently). Sparse path interactions could break, mitigated by dense-only smoke first.
- **Effort**: ~3 days per survey (mostly verification, not new code).
- **Probability of success**: MEDIUM-HIGH — Blackwell SM120 has native FP4 tensor support, plumbing surveyed.
- **Reference**: `SURVEY_NVFP4_KV_P1_20260428_1130.en.md`, `PROPOSAL_iteration_M20_kv_fp4_ablation.{en,zh}.md` (untracked files in your tree — may already exist as drafts).

### #3 — FP8 W8A16 weight quantization (the BIG lever, but heavy)
- **Change**: replace GPTQ-Marlin W4A16 with FP8 W8A16; native Blackwell FP8 tensor cores at **296 TFLOPS** vs Marlin's effective ~150 TFLOPS for W4 dequant+FP16 GEMM (~2× theoretical).
- **Why eventually**: **GEMM is 85.3% of prefill.** Directly attacks the actual bottleneck. Top-1 FlashSALA at 97.47 score almost certainly uses something like this (Marlin-tuned-or-replaced).
- **Expected gain**: 10–30% prefill (catalog estimate); plausibly **transformative** for both S1 and Smax.
- **Risk**: medium-high — full quantization pipeline rewrite (preprocess_model.py path), needs FP8 GEMM kernel in sgl-kernel that's tuned for SM120, accuracy regression risk is ~8-bit-typical (low).
- **Effort**: HIGH — 1–2 weeks. Includes: FP8 quantization in preprocess (probably via existing `compressed-tensors` or torchao path), sglang loader path, sgl-kernel FP8 GEMM verification on SM120, accuracy & speed testing.
- **Probability of success**: MEDIUM — depends on whether sglang already has a clean FP8 W8A16 path for our model arch (need to verify); if it does, this drops to "medium effort."
- **Note**: distinct from the **W4-FP8 CUTLASS spike** (which IS parked — only ~5–12% for 3–4 weeks of CUTLASS work). W8A16 is a different bet.

---

## Recommended sequence

1. **Round 1 (this week)** — Candidate #1 quick-win test: 1 fcloud round, ~3h total. If it gains, ship as v21 immediately.
2. **Round 2** — Candidate #2 NVFP4 KV: ~3 days of focused work + acc/speed validation. Ship if accuracy holds.
3. **Round 3** — Decision point: pursue #3 FP8 W8A16, OR investigate FlashSALA's Marlin-tile-tuning approach (if any leaked details).

This stages risk: free config win first (#1), proven-plumbing memory win (#2), then the big GEMM rebuild (#3) only after we've harvested the cheap gains and have a clearer picture of the gap.

---

## Question for you

Pick one of:
- **(I)** Approve sequence 1→2→3, start with #1 — I draft a one-page proposal for the Tier 1 retest, you approve, then we run.
- **(II)** Skip ahead — start at #2 (NVFP4 KV) since survey/plumbing is already done.
- **(III)** Skip to #3 (FP8 W8A16) — go big.
- **(IV)** Different priority altogether — name what you want.

I lean **(I)** — the Tier 1 retest is so cheap it would be malpractice to skip without a measured result on the long-context dataset.             

