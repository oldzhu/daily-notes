Compacted conversation

Read [](file:///home/oldzhu/sglang/.github/copilot-instructions.md#70-70), lines 70 to 110

Replacing 6 lines with 30 lines in [](file:///home/oldzhu/sglang/.github/copilot-instructions.md)


```

```


Read memory

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_local_vs_official_dataset.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add .github/copilot-instructions.md && git commit -m "docs(copilot-instructions): add 'Local vs Official evaluation' section

Record key facts that must be consulted whenever analyzing local vs official
result gaps:

1. Accuracy datasets differ — local uses only public perf_public_set.jsonl;
   official uses public + private. Private is undisclosed but likely similar
   format/distribution. Safety margin: aim local >= 80% to stay above 77%.

2. Speed benchmark datasets differ — local speed_{s1,s8,smax}.jsonl is
   hand-built; official is undisclosed and was recently updated with more
   long-context samples. Local ratios do NOT predict official. Prioritize
   long-context + high-concurrency optimizations.

3. Accuracy-drift anomaly — v18-A/B/C (same package) scored 78.71/80.51/76.64.
   Never trust a single official number; require >=2 resubmits and plan to
   the worst of the batch." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -4

Done. Committed `b01cdf4ee`; persisted to memory at soar_2026_local_vs_official_dataset.md.

## My take on your strategic question

**Yes — raising the local accuracy floor is the right priority, and it stacks perfectly with Iteration A-0.** A few thoughts:

**1. Why a local margin helps even though the public set is "the same":**
- Official = public + private. If private is similar format but contains samples where our GPTQ/FP8 quantization error happens to bite harder, we could lose 1–3 pts just from dataset mix.
- Even with mcq runaway fixed, fcloud contention during official eval can still introduce random timeouts → a few mcq returning empty → another 1–2 pts lost.
- Total worst-case gap local→official: **3–5 pts**. Local 80% → official 75%. So local 80% is NOT safe yet. Target should probably be **local ≥ 82%** to have C ≥ 0.92 guaranteed.

**2. Why Iteration A-0 is the highest-leverage accuracy fix we have in hand:**
- Test 34a local mcq = 56.67% (one task alone). If we fix runaway and mcq rises to 80% (still well below the 96% niah gets), **overall local accuracy rises from ~78% to ~82%** from a single task.
- mcq is the task with the widest variance across our tests (Test 29: 96%, Test 32: 40%). The high-end shows the model CAN answer correctly — the low-end is purely "thinking overflowed max_tokens before answer." Fixing that is converting lost points back, not earning new ones.
- So Iteration A-0 is simultaneously: (a) unlocks C ≠ 0, (b) buffers against private-set variance, (c) fixes Smax by 2–5× officially. Three wins from one fix.

**3. About the strange v18-A/B/C drift on the same package:**
- If organizers are truthful that accuracy data is unchanged, the 78.71→80.51→76.64 drift must come from: (i) stochastic sub-sampling of private set per batch, or (ii) harness load causing timeouts. Runaway mcq amplifies both — the 10,946-token chains are the thing most likely to hit deadlines.
- The earlier pre-v18 package you resubmitted will be very informative: if it shows similar 3–5 pt drift, then "multi-run averaging" is a general property, and we must **always plan to the worst**.

**4. Recommendation for next steps (in priority order):**
1. Wait for your earlier-package resubmit result to confirm the drift is a general pattern, not specific to v18.
2. Execute **Iteration A-0 Layer 1** (eval-script per-task `max_out_len` + `</think>` stop) — gives us fast local validation that the diagnosis is correct. Can be done without touching the model or server.
3. Execute **Iteration A-0 Layer 2** (chat template or `--reasoning-parser qwen3` or generation_config EOS) — the only layer that moves the official number.
4. Resubmit v19 **twice** (following our "≥2 submissions before conclusion" rule) to measure the post-fix drift band honestly.

If you agree, I can start Phase 1 of A-0 (read-only fcloud inspection of chat_template + 5 mcq prediction samples from Test 34a) as soon as you start the instance. 



