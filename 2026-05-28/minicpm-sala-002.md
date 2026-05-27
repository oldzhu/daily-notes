Starting: *Read CHANGE_0154 §4 Stage 2 spec* (1/8)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0154_medusa_phase_r1b_design.en.md)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_info.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm_medusa_heads.py)

Searched for regex `is_ngram\(\)` (`**/python/sglang/srt/**`), 16 results

Searched for text `class NgramWorker` (`**/python/sglang/srt/**`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/base_spec_worker.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#200-200), lines 200 to 255

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/spec_info.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py#270-270), lines 270 to 925

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#1690-1690), lines 1690 to 1880

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py#855-855), lines 855 to 880

Searched for regex `create_worker\(|NGRAMWorker\(` (`**/python/sglang/srt/**`), 5 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py#520-520), lines 520 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py#470-470), lines 470 to 520

Searched for regex `SOAR_SPEC_MEDUSA|speculative` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 20 results

Searched for regex `medusa|MEDUSA|speculative_num_medusa` (`**/python/sglang/srt/server_args.py`), 17 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#2215-2215), lines 2215 to 2260

Now I have full context. Stage 2 is **simpler than the design doc suggested** — pure delegation only, no `capture_hidden_mode` plumbing, no `_dummy_run`/`cuda_graph_runner` edits needed (MEDUSA stays in normal DECODE mode). Only 2 files change.

Let me implement: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#1-1), lines 1 to 10

Ran terminal command:  cd /home/oldzhu/sglang && rm python/sglang/srt/speculative/medusa_worker.py

Created [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)

Now patch scheduler.py for the disaggregation safety guard:Autopilot recovered from a request error

Replacing 2 lines with 8 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py)


```

```


Checked [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py), [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler.py), no problems found

Ran terminal command: import ast; ast.parse(open('python/sglang/srt/speculative/medusa_worker.py').read()); ast.parse(open('python/sglang/srt/managers/scheduler.py').read()); print('OK')

Now write CHANGE_0155 docs + chat log. 

Ran terminal command:  date +%Y%m%d_%H%M

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.zh.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_medusa-r1b-stage2_20260511_0804.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_medusa-r1b-stage2_20260511_0804.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add python/sglang/srt/speculative/medusa_worker.py python/sglang/srt/managers/scheduler.py docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.en.md docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.zh.md docs/soar_2026_changes/chat/CHAT_medusa-r1b-stage2_20260511_0804.en.md docs/soar_2026_changes/chat/CHAT_medusa-r1b-stage2_20260511_0804.zh.md && git status --short

Ran terminal command:  cd /home/oldzhu/sglang && git commit -m "feat(medusa): R1b Stage 2 - heads-shadow smoke-test worker (CHANGE_0155)

Stage 2 of CHANGE_0154 Medusa Phase R1b. Replaces the R1a NotImplementedError
stub with a pure-delegation MedusaWorker that:

  * Accepts the scheduler's draft_worker_kwargs signature
  * Instantiates MedusaHeads(K, hidden, lm_head, dtype) on GPU
    (~32 MiB BF16 at K=1, hidden=4096) using the target model's lm_head
  * Validates target_model.lm_head exists with a clear error message
  * Delegates forward_batch_generation byte-identically to target_worker
    (no capture_hidden_mode mutation, no forward_mode change, no draft
    tokens, no spec_info)
  * Logs 'MedusaWorker Stage 2 ready' summary at init
  * Reports num_accepted_tokens=0 (no speculation yet)

Also adds a defensive 'or is_medusa()' guard at scheduler.py L866 in the
init_disaggregation branch that skips draft_token_to_kv_pool (Medusa has
no separate draft model, mirroring NGRAM). Single-instance SOAR does not
hit this path but the guard keeps the contract correct.

The 5 other is_ngram() call sites in model_runner.py / cuda_graph_runner.py
intentionally NOT modified because Stage 2 stays in normal DECODE mode and
never triggers TARGET_VERIFY capture. Stage 3 will revisit those when verify
forwarding lands.

Net code change: 2 files (worker rewrite + 1-line scheduler guard).

Stage 2 pass criteria for fcloud:
  * Server boots with SOAR_SPEC_MEDUSA=1
  * Log contains 'MedusaWorker Stage 2 ready'
  * Accuracy byte-identical to baseline
  * S1 regression <= 3%

Stage 3 (real verify+rewind) is hand-off documented in CHANGE_0155 sec 8.

Refs: CHANGE_0153 (R1a), CHANGE_0154 (R1b design + Stage 1 helpers)." && git push minicpm-src mixed_minicpm_cudagraph

Stage 2 pushed: `c5de9b1bd..e3b65fc01`.

**What's in this commit** (`e3b65fc01` — "feat(medusa): R1b Stage 2"):

| Change | File | Lines |
|---|---|---|
| MedusaWorker rewrite (delegation + heads alloc) | medusa_worker.py | full rewrite, ~150 LOC |
| `init_disaggregation` guard | scheduler.py | +5 / −1 |
| Design doc EN | CHANGE_0155_medusa_phase_r1b_stage2.en.md | new |
| Design doc ZH | CHANGE_0155_medusa_phase_r1b_stage2.zh.md | new |
| Chat log EN/ZH | CHAT_medusa-r1b-stage2_20260511_0804.en.md | new |

**Behavior** with `SOAR_SPEC_MEDUSA=1`:
- Server boots, log shows `MedusaWorker Stage 2 ready: K=1, hidden=4096, dtype=torch.bfloat16, device=cuda:0, approx_weight_MiB=32.0`
- Decode path = pure delegation to target worker → output should be **byte-identical** to baseline
- No verify, no snapshot, no `capture_hidden_mode` change

**Ready for fcloud validation.** When you approve, the workflow is:
```
start-instance → sync → restart-server (SOAR_SPEC_MEDUSA=1)
  → wait-server → accuracy → speed --variant s1
  → server-logs (verify "Stage 2 ready") → pause-instance
```

Pass criteria: accuracy byte-identical to baseline (≈79.29% ori_acc) + S1 regression ≤ 3%. If both pass, Stage 3 (real verify + rewind) is unblocked.  



