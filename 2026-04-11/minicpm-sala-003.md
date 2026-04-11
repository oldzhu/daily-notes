Good question. Let me review the evidence carefully.

**Were there timeouts?** No — the eval completed all 150 items successfully. The `call_sglang_api` has a per-request timeout of 3000s (50 min), and individual requests took at most ~74s (the slowest item #86). The summary.json shows all 5 tasks × 30 items = 150 total correctly counted. No timeout errors.

**The accuracy drop is genuine.** The key clue is in the output token stats from the summary:

| Task | FP8 avg output tokens | Expected completion tokens | Accuracy |
|------|-----------------------|--------------------------|----------|
| niah | **13,532** | 163–702 | 36.67% |
| qa | 107 | 69–151 | 33.33% |
| fwe | 9,474 | 480–1,256 | 97.78% |

NIAH's avg output is **13,532 tokens** — about 20× more than expected. The model with FP8 KV can't find the needle, so it generates long reasoning chains and still gets wrong answers.

**Why FP8 KV hurts sparse attention on long context specifically:**

1. **Sparse scoring relies on KV precision** — InfLLMv2 computes attention scores on K1/K2 representations to rank ~2000 blocks (128K/64) and select top-96. FP8 quantization noise distorts these rankings.

2. **NIAH is the most sensitive** — finding one specific "needle" among 128K tokens of padding requires precise block ranking. A small FP8 scoring error can cause the wrong block to be selected, losing the answer entirely (100% → 36.67%).

3. **Short context (MCQ, 103–664 tokens) is unaffected** — with ~10 blocks total, sparse selection barely activates, so FP8 noise doesn't matter (66.67% in both FP8 and bf16 tests).

4. **FWE is robust** — fill-word extraction likely has more distributed answer signals across many blocks, not a single needle.

This aligns perfectly across all our tests: **every sparse+FP8 config shows the same NIAH collapse** (36.67% in Tests 1, 5, 11), while bf16 KV preserves NIAH (96–100% in Tests 3, 8b, 9). The conclusion stands: FP8 KV is incompatible with sparse attention for long-context retrieval tasks.
