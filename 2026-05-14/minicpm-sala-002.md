**Yes for accuracy-eval wall-clock, but the answer is more nuanced for the official S1/S8/Smax speed benchmark.** Let me split the two cases.

### 1. Accuracy-eval wall-clock duration: YES, big drop

This is the `benchmark_duration` reported when running eval_model_001.py against `perf_public_set.jsonl`. mcq currently spends ~10,946 output tokens per sample × ~30 mcq samples = ~330k tokens of pure waste. With thinking off:

| Metric | Now (thinking on) | After Phase 14.1 (mcq thinking off) |
|---|---|---|
| mcq avg_out tokens | ~10,946 | ~10–30 |
| mcq total decode time per run | ~70–80% of total run time | ~0.1% |
| Estimated full eval wall-clock | 2,800–3,000s | **800–1,200s** (Iter A-0 estimate) |

So local accuracy runs become ~3× faster, which means **iteration cadence improves** (4-run quartet drops from ~3h to ~1h). That's an internal productivity gain.

### 2. Official S1/S8/Smax speed benchmark: NO direct effect

The speed benchmark is run against a different dataset:
- **Official**: their private speed dataset (longer-context biased, see copilot-instructions.md notes).
- **Local**: `/root/data/speed_{s1,s8,smax}.jsonl` — hand-built by us, NOT the official set.

Neither contains mcq queries. The speed datasets are dominated by long-context tasks (more like cwe/qa/niah patterns). Disabling mcq's thinking does nothing to those samples because:
- The chat template only flips `enable_thinking=False` when the input matches the mcq pattern (lettered choices A./B./C./D.).
- Speed-benchmark inputs don't match → thinking stays ON → identical behavior to v20 baseline.

So **S1/S8/Smax should be unchanged within noise** — which is exactly what we want for the submission cadence rule (don't submit until speed↑ AND acc holds).

### 3. Indirect effects to verify

A few subtle interactions worth checking when CHANGE_0140 lands:

| Risk | Probability | Mitigation |
|---|---|---|
| Chat-template patch accidentally affects non-mcq inputs (false-positive detection) | Low | Detection predicate must be tight (require explicit "A.", "B.", "C.", "D." line pattern); validate by re-running speed benchmarks once before submitting. |
| GPTQ Marlin / fused-QK-norm-RoPE / mixed-chunk codepath has hidden coupling to thinking-mode token-emission patterns | Very low | Speed test will catch it. |
| Chat-template change forces tokenizer regeneration which changes some IDs | Very low | The Jinja change is data-only, not vocab; safe. |

### Bottom line

- **Local eval cadence**: ~3× faster (productivity win).
- **Official S1/S8/Smax**: unchanged (correct — that's the design).
- **Official Performance Score**: unchanged (because speed benchmark unchanged).
- **Official Correctness Coefficient C**: should rise (mcq accuracy lifts mean, may cross C-tier boundaries 0.92 → 0.96 or 0.96 → 1.0).
- **Final Score = Performance × C**: rises proportional to C improvement.

So Phase 14.1 is a **pure C-multiplier play** with a free productivity bonus. It does not interfere with the speed-improvement track. The two tracks (Phase 14 accuracy + ongoing speed work) are orthogonal and can be developed in parallel — exactly why the submission cadence rule says "submit only when both improve simultaneously": Phase 14.1 alone earns no submission, but bundled with the next speed win it becomes the multiplier.   


