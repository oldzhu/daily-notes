Searched for regex `attention-backend|minicpm_flashinfer|flashinfer` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 20 results

You're right — I got the default wrong. Let me re-check the actual launch path.  

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#300-300), lines 300 to 395

**You're absolutely right — I was wrong.** Default is `SOAR_BACKEND_VARIANT=flashinfer` → stock flashinfer attention backend. The `NotImplementedError` lives in `minicpm_flashinfer` (custom), but our v22 / v23 baseline launches with **stock flashinfer**, which fully supports `TARGET_VERIFY` (eagle/ngram use it everywhere).

This is in fact the **opposite of a blocker** — it means the de-risk is already done by the fact that stock flashinfer is well-trodden for spec decoding. Let me revert the §11 panic and proceed with Stage 3a.  

Multi-Replace String in Files

Replacing 35 lines with 11 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.en.md)


```

```


Replacing 35 lines with 11 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.zh.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.en.md docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.zh.md && git commit -q -m 'docs(medusa-r1b): correct §11 — default backend is stock flashinfer (supports TARGET_VERIFY)' && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Read memory

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/soar_backend_variant_default.md)

Correction pushed (`580d86582`). Memory note logged so I don't repeat this.

**Status**: Stage 3 is **not blocked**. The original §1-10 plan stands. Stock flashinfer (our default) has full TARGET_VERIFY support — eagle and ngram rely on it.

**Proceeding to Stage 3a code now.** Plan:
1. Replace medusa_worker.py Stage 2 pass-through with Stage 3a draft+verify flow.
2. Local syntax/import sanity check.
3. Ask before kicking fcloud test.

Working on the code:  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_info.py#1-1), lines 1 to 75

Before I commit to writing ~200 LOC of careful Stage 3a code, I want to suggest a cheap de-risk that I should have done before any of this proposal cycle:

**Run NGRAM (`SOAR_SPEC_NGRAM=1`) on fcloud first.** This is:

- **Zero new code** — the flag is already wired in prepare_env.sh (line 441).
- **Tests the entire verify pipeline** on our exact config (stock flashinfer + GPTQ + FP8 KV + dense + mixed-chunk + torch.compile + cuda-graph).
- **30-min fcloud cost** (sync + restart + accuracy run).

If NGRAM accuracy comes back ≈ 80% → the verify infra is healthy end-to-end, and Stage 3a becomes a "swap n-gram cache lookup → MedusaHeads forward" job with all the hard bits (KV alloc, tree mask, attention metadata, hidden capture, batch result extraction) already proven on our config.

If NGRAM fails or crashes → we save myself writing Medusa code on top of a broken pipeline, and we get a precise error pointing at exactly where the verify infra breaks.

**Tradeoff**:

| Path | Time to first signal | If Stage 3a writes break | If verify pipeline broken |
|------|----------------------|--------------------------|---------------------------|
| Code 3a first, then test | ~2 hrs code + 30 min fcloud | hard to tell if it's head code or verify infra | both costs eaten |
| Probe NGRAM first | 30 min fcloud | n/a | save the ~2 hrs of speculative code-writing |

**Recommendation**: probe NGRAM first. **OK to run?** If yes, I'll: start fcloud → flip `SOAR_SPEC_NGRAM=1` → restart server → accuracy run → analyze → pause fcloud. Then we make the Stage 3a code decision with data, not guesses.   



