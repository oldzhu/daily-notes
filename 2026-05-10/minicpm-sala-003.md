that is wrired, although we have the dense path baseline so far is the best, I still don't want to give up on the sparse line path, as the 1st original model which we try to improve in the competition is non-quanted model which has mixed 8 sparse attetion layers with 24 lighting attention layers, by default sparse is open?  At least, we didn't see so many read timout or long time when testing the original model.
let us disucss more on this before decide to discrad or continue:
from the page https://soar.openbmb.cn/toolkit
could see the below default SGLANG args could be overwrited by ourselves setting.
...
SGLANG_SERVER_ARGS	--disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768
若设置则覆盖默认（建议单引号包住）
...
注意事项：

SGLANG_SERVER_ARGS 里请用连字符参数名：例如 --dense-as-sparse，不要写 --dense_as_sparse（镜像不做下划线自动转换）。
...
I saw --attention-backend flashinfer would that mean by default it is not minicpm_flashinfer? and I also could see force-dense-minicpm is not used by default.
another thing is that I noticed in the testing, in the eval script command line, you pass the model path to the model path to the gptq quanted model, not the path to the original non-quanted model, would that cause any problem?
========
Excellent observations. Both points are real and the second one is potentially **critical**. Let me investigate before we conclude anything. 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#1-1), lines 1 to 120

Searched for regex `model_path|MODEL_PATH|eval_model` (`**/scripts/fcloud/fcloud_workflow.py`), 20 results

Searched for regex `step_accuracy|--model_path` (`**/scripts/fcloud/fcloud_workflow.py`), 5 results

Confirmed both concerns are real. Let me address them:

## Your point #2 (eval `--model_path` mismatch) — this is potentially the cause

eval_model_001.py uses `--model_path` to load the **tokenizer + GenerationConfig (eos_token_id, stop words) locally**. It does NOT tell the server which model to use (server uses whatever was launched). 

In our Round 13e Test 1, server ran `/root/models/openbmb/MiniCPM-SALA` (non-quanted) but eval loaded tokenizer/eos from `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` (GPTQ).

**If GPTQ model dir has different `tokenizer_config.json`, `chat_template`, or `generation_config.json` from the non-quanted dir, the prompts/stop-tokens are wrong** → could cause runaway generation hitting `max_tokens=65536` → exactly the timeout symptom we saw. Even if the tokenizer files are identical (likely, since GPTQ derives from non-quanted), this is still a correctness bug we must fix.

## Your point #1 (default `--attention-backend flashinfer`, no `--force-dense-minicpm`) — also significant

Official toolkit default: `--disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768`. So:
- **Default is sparse-on** (no `--force-dense-minicpm`).
- **Default backend is generic `flashinfer`, not `minicpm_flashinfer`.**

Our Round 13e used `--attention-backend minicpm_flashinfer`. The custom `minicpm_flashinfer` backend is exactly the path with the over-fill bug we just patched (and apparently a deeper sparse-attn cost we have not yet identified). The original (organizer-supplied) non-quanted model running under default args clearly works in their environment — so either:
- (a) `--attention-backend flashinfer` routes the sparse layers through a different (working) code path, or
- (b) it dispatches them via the linear-attn hybrid backend and falls back to dense for sparse layers, or
- (c) the official baseline is also slow but tolerated because their accuracy dataset is shorter/different.

Let me verify (a/b/c):   

Searched for text `attention-backend` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 3 results

Searched for regex `minicpm_flashinfer|register_attention_backend|attention_backend ==` (`**/python/sglang/srt/layers/attention`), no results

Searched for text `minicpm_flashinfer` (`**/python/sglang/srt`), no results

Searched for text `minicpm_flashinfer`, 20 results

OK, I have the picture now. Let me synthesize:

## Summary of investigation

### Your point #1 — answer
- Official default `--attention-backend flashinfer --chunked-prefill-size 32768` (no `--force-dense-minicpm`) is the **organizer's baseline**. CHANGE_0070 doc confirms: `--force-dense-minicpm` internally rewrites `minicpm_flashinfer` → `flashinfer`. That means the **official default (`--attention-backend flashinfer`) does NOT activate the model's native sparse routing** — the 8 sparse layers are dispatched as regular attention. It is effectively dense. That's why "the original model with default args ran with no timeouts" — they were running effectively-dense too.
- Sparse routing is ONLY active under `--attention-backend minicpm_flashinfer` (a custom backend implemented in this repo, not in upstream sglang). That's the path with the over-fill we patched and (apparently) some other long-context cost.
- **Implication**: there is NO already-working sparse decode at concurrency=32 in the toolkit baseline either. The custom `minicpm_flashinfer` is the only sparse path, and it's the one we need to fix.

### Your point #2 — confirmed bug
`step_accuracy` in fcloud_workflow.py hardcodes `--model_path {MODEL_PATH}` (GPTQ path) regardless of `--quant-mode`. For the noquant test, eval loaded:
- tokenizer from GPTQ dir
- `GenerationConfig.eos_token_id` from GPTQ dir
…while server ran the non-quanted model. If those files differ between the two dirs (very possible — `preprocess_model.py` may modify `chat_template.jinja`/`generation_config.json` for GPTQ), the prompts/stop-tokens are wrong → could trigger runaway generation hitting `max_tokens=65536` → exactly the timeout symptom.

This is a real bug regardless of #1, and could partially or fully explain the Round 13e timeouts.

## Recommended next steps (proposal-only, awaiting approval)

1. **Fix the eval-script model_path mismatch** (file: fcloud_workflow.py):
   - Make `step_accuracy` accept `quant_mode` and select the matching `*_MODEL_PATH` for `--model_path`, just like `step_restart_server` does.
   - Update both `cmd_accuracy` and `cmd_full` to forward `quant_mode`.
2. **Diagnostic step before re-testing**: on next fcloud session, before any test, diff the tokenizer/template/eos files between the two model dirs:
   ```bash
   diff /root/models/openbmb/MiniCPM-SALA/tokenizer_config.json \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/tokenizer_config.json
   diff /root/models/openbmb/MiniCPM-SALA/generation_config.json \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/generation_config.json
   diff /root/models/openbmb/MiniCPM-SALA/chat_template.jinja \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/chat_template.jinja  2>/dev/null || true
   ```
3. **Re-run Round 13e Test 1** with the fixed model_path (CHANGE_0133 already applied). If timeouts disappear → tokenizer mismatch was the cause; sparse path may be viable after all. If they persist → sparse decode at long-context is genuinely too slow and we move on.
4. **Optional follow-up**: also test with the *exact* official baseline args (`--disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768`, no `minicpm_flashinfer`, no `--force-dense-minicpm`) to characterize what the "out-of-the-box" toolkit speed actually is — useful as a sanity floor.

**Scope note on rule freshness**: this re-confirms the official toolkit page; default backend is `flashinfer` (non-sparse). I'll add a note that sparse routing requires the custom `minicpm_flashinfer` backend.

Want me to proceed with step 1 (fix the eval-script model_path bug) now, then we discuss step 2/3 before any new fcloud run?Continued with Autopilot: Task not yet completePicking up where I left off. The user asked for a discussion before deciding, and I responded with analysis + a question ("Want me to proceed with step 1 ... ?"). That's a deliberate awaiting-approval state per repo rules ("Ask for approval before each code modification"). Calling task_complete now to honor the hook.  



Awaiting your call on whether to proceed with step 1 (eval-script `--model_path` fix) before any further fcloud run.
