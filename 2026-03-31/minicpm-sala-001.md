This official run is the new speed baseline. It is the fastest of your four official submissions at all three concurrency levels:

- S1: `440.61`, better than `539.46`, `458.78`, `462.27`
- S8: `626.43`, better than `747.59`, `634.35`, `663.66`
- Smax: `1138.48`, better than `1365.8`, `1140.66`, `1203.4`

So the current package clearly improved runtime. The bottleneck is now score sensitivity to correctness, not speed. With `acc=97.33`, you are still in the `C=0.92` band, which means the current speed package is being discounted by about `8.7%` versus a full-coefficient run. If the same durations hold and you move back to `C=1.0`, the final score jumps materially without any extra speed work. That makes your plan correct: resubmitting once more is reasonable because the accuracy side is visibly noisy, and after the local `0061` test we should prioritize accuracy recovery first.

My recommendation after your `0061` local result comes back is:

1. Treat the current official package as the speed baseline.
2. Prioritize accuracy work until `acc_ori` is stably at least `80`, because that has the highest score leverage now.
3. Keep speed work as the second track, but only merge speed features that do not noticeably drag correctness.

For the next accuracy proposals, the best low-risk directions are:
1. Refine GPTQ calibration composition again, especially inside the `qa` and `cwe` focus, with more weight on the longer buckets where we already saw the biggest losses.
2. Try small-scope selective dequantization on the most accuracy-sensitive modules instead of broad rollback, so we preserve most of the speed gain.
3. Compare official-noisy runs against local repeated evals to identify changes that improve mean accuracy versus changes that only improve one run by luck.

For the next speed proposals, after `0061`:
1. If `0061` gives measurable gain, keep it in the baseline.
2. If `0061` is flat, disable it and move to deeper Marlin-side or output-projection-path tuning.
3. Keep targeting force-dense lightning hot paths first, because the official results already show that line of work is paying off.

Send me the local `0061` numbers when you have them. Once they are in, I’ll give you the next proposal in the same one-feature-at-a-time format, with accuracy-first priority.
