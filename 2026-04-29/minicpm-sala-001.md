Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#130-130), lines 130 to 140

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_model.sh)

Searched for regex `gptq|quantize|quantization` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 30 results

Searched for regex `quant|preprocess|prepare_model` (`**/scripts/fcloud/fcloud_workflow.py`), 16 results

I'll prepare everything now (edit to Test 20 config, commit, push, and deliver the strategic proposal), then we run when fcloud is up.

**Step 1: Restore prepare_env.sh to exact Test 20 config**

Test 20 config = chunk=32K, prefill-max-req=1, sched-cons=1.0, running=24, mixed-chunk, torch.compile(bs=8). Current file has v19 scheduling (chunk=65K, prefill=4, sched=0.8) and no torch.compile. Fixing: 

Replacing 1 lines with 1 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git diff benchmark/soar/demo_sala/prepare_env.sh | head -20 && git add benchmark/soar/demo_sala/prepare_env.sh && git commit -m "test34: restore Test 20 best combination (chunk=32K, prefill-max-req=1, sched-cons=1.0, running=24, mixed-chunk, torch.compile bs=8, fp8_e5m2 KV)" && git push minicpm-src mixed_minicpm_cudagraph

Test 34 config pushed (commit `9851c0add`).

**Step 2: Re-quantization plan** — The existing model on new fcloud at `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` may be a copy from old fcloud. Re-quantizing on this SM120 instance would calibrate against current CUDA/Triton/kernel versions. The official command pattern:

```bash
cd /root/submission_sim
SOAR_QUANT_MODE=gptq bash prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA-Copy \
  --output /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8-fresh
```

This takes ~30-60 minutes on SM120 (GPTQ calibration over 8 samples). When complete, we point server at the fresh model via `--model-path` override.

---

# Strategic Proposal: Next 3 Iterations to Reach Top 5/Top 3

## Evidence summary (so far)

| Finding | Evidence | Implication |
|---|---|---|
| **Speed is our weakest link** | Leaderboard: team-beta=39.62 vs #5≥79.55 → need **2× performance score** | Speed > Accuracy (currently C=0.96 reliably, speed in bottom half) |
| **Accuracy floor ≈77-79% on new fcloud** | Tests 29/30/32/33: no single knob moves the mean | Chasing acc via config is futile; need structural change |
| **Per-task variance ±20-40pt** | Test 29 mcq=96 vs Test 30 mcq=53 with one flag flip | 150-sample concurrency=32 eval is intrinsically noisy |
| **torch.compile ~0% speed, +3min boot** | Test 33 vs 29: 3016s vs 3005s duration | Boot-time cost is real in quant+eval ≤5h budget |
| **Scoring formula** | `Final = (S1×0.4 + S8×0.3 + Smax×0.3) × C` | 10% speed gain = +10 score pts; 1 C-tier gain (0.92→1.0) = +8.7% multiplier |

## My opinion on "acc vs speed as next goal"

**Accuracy as primary goal is the WRONG choice** for 3 reasons:

1. **Diminishing returns**: C=0.96 → C=1.0 = 4.2% score bonus. A 10% speed win = 10% score bonus. A 15% speed win = 15%. Speed dominates.
2. **Leaderboard gap is multiplicative in speed**: #5 team is ~2× our score. Even perfect C=1.0 gets us to ~41 × (1/0.96) = 42.7 → still 47% short of #5. **Only speed can close this**.
3. **Accuracy we have is already at the noise ceiling**: 77-79% local = C=0.96. To reliably hit C=1.0 we need >82% normalized, which means ~81% acc_ori — that's a model-capability ceiling, not a config tuning target.

**Accuracy work IS worthwhile only** if: (a) the v18 resubmit result shows we're consistently near the C=0 cliff (<78% acc_ori), or (b) we have a specific accuracy-bottleneck task (like the runaway mcq chains) that can be fixed structurally. Otherwise, speed wins.

## Proposed 3-iteration roadmap

### Iteration A — Speed: Marlin GPTQ kernel specialization for SM120 decode-heavy shapes (HIGH impact)
- **Observation**: decode TPS = 439 (Test 29). BF16/FP16 peak on SM120 is 148 TFLOPS. Marlin GEMM at W4A16 decode (batch=1-8) currently underutilizes tensor cores on small M.
- **Action**: extend the SM120 Marlin tile table (CHANGE_0125) with decode-optimized tiles (M=1,2,4,8 × K=7168/4096 × N=accordingly) + add `num_threads` specialization for the smaller M cases. Reference: vllm and TensorRT-LLM W4A16 kernels for SM100/120.
- **Expected gain**: +5-15% decode TPS → S1/S8 reduce by ~5-15%. Biggest win on S1 (pure decode).
- **Risk**: kernel build time (first time ~4h; incremental ~3min). Accuracy neutral (tile choice doesn't change math).
- **Effort**: 1-2 iterations, each compiles + runs speed eval. Documented in CHANGE_0140_sm120_decode_marlin_tiles.

### Iteration B — Speed: Disable torch.compile + use CUDA graphs directly (MEDIUM impact)
- **Observation**: Test 33 proved torch.compile adds 3min boot with ~0% speed gain at eval concurrency. The 5h budget (incl. quantization+eval) is tight; saving 3min per server restart × multiple restarts during official eval = measurable.
- **Action**: remove `--enable-torch-compile --torch-compile-max-bs 8`; verify CUDA graph capture (--cuda-graph-max-bs or similar) still works and covers decode paths.
- **Expected gain**: +3-5min per restart, no runtime speed change. Pure budget savings.
- **Risk**: very low; Test 33 already ran without torch.compile and finished in 3016s.
- **Combo**: apply after Iteration A lands, bundle into v21 submission.

### Iteration C — Mixed accuracy/speed: Speculative decoding with **trained** draft (HIGH impact if successful)
- **Observation**: prior EAGLE3 attempt (CHANGE_0090) failed due to untrained draft → accept_rate=0.26. A properly trained draft could give 2-3× decode throughput.
- **Action**: two options (both high-risk but potentially game-changing):
  - **C1 (faster path)**: use n-gram speculative decoding (no model training needed) — token-pattern match from prompt/prior generation. Ships in sglang. Works best on tasks with repetitive structure (NIAH, FWE).
  - **C2 (bigger upside, bigger effort)**: train a 1-layer EAGLE draft on the provided calibration data. Requires ~1-2h training on SM120.
- **Expected gain**: n-gram → +10-30% on NIAH/FWE; EAGLE(trained) → +50-100% on all decode.
- **Risk**: accept_rate can tank if draft is wrong; accuracy must stay >97% normalized. Needs rollback safety.
- **Effort**: C1 = 1 iteration (config flag + eval); C2 = multi-day training + iteration.

## Recommended sequencing (if Test 34 confirms best-config reproducibility)

1. **First** (today): Test 34 Test-20 reproduction + re-quant comparison → establish clean baseline on new fcloud
2. **Next**: Iteration A (Marlin SM120 decode tiles) — highest expected gain, cleanest accuracy risk profile
3. **Parallel**: submit v20 (= Test 34 config package) as safety net if it beats v18 resubmit
4. **After A lands**: Iteration B bundled into v21
5. **After B**: Iteration C1 (n-gram) as fast exploration; C2 (EAGLE training) as stretch goal

## Key decision point awaiting your input
- **v18 official resubmit result** — if it lands near 78-80% (C=0.96) with current rank still far from #5, this confirms speed-first strategy
- **If v18 lands below 77% (C=0)** — accuracy work becomes urgent; then we probably need C2 accuracy structural fix (different KV cache layout, or recalibrated GPTQ with seed sweep)

---

**Ready to run once fcloud starts**:
```bash
# Test 34a (quick, existing model):
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server --quant-mode gptq
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
# If we also want speed benchmark, do it before shutdown

# Test 34b (re-quant, slower):
# On fcloud:
cd /root/submission_sim
SOAR_QUANT_MODE=gptq bash prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA-Copy \
  --output /root/models/openbmb/MiniCPM-SALA-fresh-gptq
# then restart-server --model-path that path, accuracy+speed
```

Tell me when fcloud is up and which order you want (34a first, or re-quant first, or both).   



