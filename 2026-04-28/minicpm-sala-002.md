Ran terminal command:  cd /home/oldzhu/sglang && cat benchmark/soar/demo_sala/prepare_env.sh 2>&1 | head -60

Ran terminal command:  cd /home/oldzhu/sglang && grep -nE "max_new_tokens|max_tokens|task|mcq" benchmark/soar/demo_sala/eval_model.py 2>/dev/null | head -40; echo "---"; grep -nE "max_new_tokens|max_tokens|task.*mcq|if task|task_type" benchmark/soar/demo_sala/eval_model_001.py 2>/dev/null | head -40

Ran terminal command:  cd /home/oldzhu/sglang && sed -n '120,170p' benchmark/soar/demo_sala/eval_model.py

Ran terminal command:  cd /home/oldzhu/sglang && grep -nE "max_out_len|generate.*task|run_dataset" benchmark/soar/demo_sala/eval_model.py | head -20; echo "---find call sites---"; grep -nE "task.*mcq|mcq.*task|if task|task in|max_out" benchmark/soar/demo_sala/eval_model.py | head -30

Now let me check one more thing — the eval's max_out_len is hardcoded to 65536, meaning we can't cap it from the submission side. But this reshapes my proposal toward KV-dtype and scheduling fixes (which we CAN control). 

Ran terminal command:  cd /home/oldzhu/sglang && grep -nE "kv.cache.dtype|SGLANG_SERVER_ARGS|prefill-max-requests|schedule-conservativeness|chunk|torch-compile" benchmark/soar/demo_sala/prepare_env.sh | head -20

# Proposal + Test Plan: Step B — Accuracy Stability Fixes

## Background

| | v18 official | v19 official | Test 29 local |
|---|---|---|---|
| acc_ori | 78.71/80.51 | **76.0/76.67** ❌ | 78.73 |
| C | 0.96/1.0 | **0 (eliminated)** | 0.96 |
| KV dtype | fp8_e5m2 | fp8_e5m2 | fp8_e5m2 |
| prefill-max-req | 1 | **4** | 4 |
| sched-cons | 1.0 | **0.8** | 0.8 |
| chunk | 32768 | **65536** | 65536 |
| torch.compile | on | on | on |

**Critical finding**: the official eval script hard-codes `max_out_len=65536` for all tasks (eval_model.py line 339). We **cannot** cap mcq generation from the submission side. The only controllable variables are **server args** in `prepare_env.sh`.

**Working hypothesis (unified)**: FP8-e5m2 KV cache loses precision on the mcq "thinking" chain, causing runaway generation (11K tokens on 270-token inputs in Tests 28/29) and misextracted answers. The aggressive scheduling in v19 (prefill-max-req=4, sched-cons=0.8) amplifies this by increasing KV cache contention/fragmentation under concurrency=32 eval load. Together these explain both the local-vs-official gap and the absolute accuracy drop.

## Rule-compliance check
- All changes are server-side config only. No model weight modification.
- KV dtype `fp8_e4m3` is an allowed SGLang option (same kernel path as e5m2).
- No forbidden mechanisms (no prefix cache re-enablement, no concurrency evasion).

## Test Matrix (single-variable bisect, ordered by priority)

| # | Change from Test 29 baseline | Hypothesis | Expected acc | Expected speed |
|---|---|---|---|---|
| **30** | `--kv-cache-dtype fp8_e5m2` → `fp8_e4m3` | e4m3 has 3-bit mantissa (vs 2), preserves KV precision for thinking chains | **79-81%** ✅ | same ±1% |
| **31** | Remove `--kv-cache-dtype` entirely (BF16 KV) | Upper bound — KV precision is the issue | 80-82% | -3 to -8% (slower) |
| **32** | Revert scheduling to v18: `--prefill-max-requests 1 --schedule-conservativeness 1.0 --chunked-prefill-size 32768 --max-prefill-tokens 32768 --max-running-requests 20` (keep fp8_e5m2, keep torch.compile) | Scheduling aggression destabilizes accuracy under concurrency=32 eval | 79-80% | -3 to -5% (S1 slower) |
| **33** (optional) | Combine winner of {30, 31} with winner of {32, v19 scheduling} | Stack best config | Best case |  |

**Test 30 is the lowest-risk, highest-signal test** — one flag change, no speed regression expected, directly tests the "KV precision hurts thinking accuracy" hypothesis.

## Decision Rules

- **If Test 30 gives acc_ori ≥ 79%** → submit as v20 (keeps v19 scheduling + e4m3 KV). Expected official acc ≈ 77-78% (above C=0.92 threshold → non-zero score).
- **If Test 30 flat (~78-79%)** → run Test 31 (BF16 KV). If BF16 gives ≥80%, we know KV precision matters; choose between e4m3 performance vs BF16 safety.
- **If Tests 30 & 31 both flat** → run Test 32. If conservative scheduling recovers acc, scheduling is the culprit; submit with less aggression.
- **If all flat** → the 2-pt local/official gap is something else (torch.compile kernel selection, concurrency-specific) — next investigation: disable torch.compile selectively or test with concurrency=32 locally.

## Exact fcloud Commands

### Test 30 (KV dtype: e5m2 → e4m3)

```bash
# Local edit (I'll do this, push to minicpm-src)
sed -i 's/--kv-cache-dtype fp8_e5m2/--kv-cache-dtype fp8_e4m3/' benchmark/soar/demo_sala/prepare_env.sh
git add benchmark/soar/demo_sala/prepare_env.sh
git commit -m "test30: swap fp8_e5m2 -> fp8_e4m3 for KV cache precision"
git push minicpm-src <branch>

# fcloud (user starts fcloud, then we run):
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server --quant-mode gptq
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
# Record result in TEST_RESULTS_TRACKING.md
python3 scripts/fcloud/fcloud_workflow.py shutdown
```

### Test 31 (BF16 KV) — only if Test 30 fails

```bash
sed -i 's/--kv-cache-dtype fp8_e4m3 //' benchmark/soar/demo_sala/prepare_env.sh
# commit + push, then sync + restart + accuracy + shutdown
```

### Test 32 (conservative scheduling) — if Tests 30+31 fail

```bash
# Change:
#   --chunked-prefill-size 65536 --max-prefill-tokens 65536 → 32768/32768
#   --prefill-max-requests 4 → 1
#   --schedule-conservativeness 0.8 → 1.0
#   --max-running-requests 24 → 20
# commit + push, run accuracy
```

## Success Criteria
- **Primary**: Test 30 acc_ori ≥ 79% locally → budget for 2-pt official drop → official acc_ori ≥ 77% → C ≥ 0.92 → non-zero score
- **Stretch**: Test 30 or 31 acc_ori ≥ 80% → official acc_ori ≥ 78% → C = 0.96 → score ~50+
- **Fail-safe**: v18-B resubmit already queued by user; v20 only replaces v18-B if strictly better

## Rollback
- Each test is a single sed+commit. To roll back: `git revert` or manual edit + push.
- If a test breaks server startup, the previous wheel + prepare_env are preserved in the fcloud submission_sim (just re-run sync to restore).

## Next-step options after Test 30 result
- **Best case** (acc ≥ 79%): submit as v20, also consider Test 31 comparison, then plan trained EAGLE3 draft for top-5 push
- **Null result**: move to Test 32 (scheduling bisect), then investigate concurrency-specific behavior
- **All negative**: shift to kernel-level accuracy work (e.g., attention backend choice, dense FlashInfer vs custom, bf16 RoPE revert)

---

**Ready to implement**: I will edit `prepare_env.sh` to swap `fp8_e5m2` → `fp8_e4m3`, commit, and push to `minicpm-src`. Then, when you start fcloud, I'll run sync → restart → accuracy → shutdown, and record Test 30 in TEST_RESULTS_TRACKING.md.

Reply with **"go"** and I'll do the edit+commit+push now (staying idle on fcloud until you start it). 


