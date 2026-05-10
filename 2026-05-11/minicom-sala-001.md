Searched for regex `dense_as_sparse|force_dense_minicpm|dense-as-sparse` (`**/python/sglang/srt`), no results

Searched for regex `dense_as_sparse|force_dense_minicpm`, 20 results

Let me verify `--dense-as-sparse` semantics: 

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#230-230), lines 230 to 410

Answer to your question first, then I'll fix and document:

**Your understanding needs a small correction.** Sparse path activates iff **both**:
1. `--attention-backend minicpm_flashinfer` (or `minicpm_flashattn`) — only this custom backend knows about sparse layers, and
2. NOT `--force-dense-minicpm` — this flag rewrites `minicpm_flashinfer → flashinfer` AND overrides `has_sparse_attention=False` at config level.

`--dense-as-sparse` does **NOT** turn sparse on by itself. It only takes effect when sparse is already active (via #1 + #2). Inside the sparse backend it sets `dense_len = 0` (vs default `hf_config.sparse_dense_len`), meaning small/short batches that would normally short-circuit to dense compute also go through the sparse code path. It's a "route more requests through sparse compute" knob, not an activator.

So default toolkit args (`--attention-backend flashinfer`, no `--force-dense-minicpm`, no `--dense-as-sparse`): condition #1 is not met → sparse layers are dispatched as plain dense attention. Now fixing the bug and documenting:  

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#320-320), lines 320 to 400

Searched for regex `cmd_accuracy|cmd_full|cmd_quick|step_quick_accuracy\(|step_accuracy\(` (`**/scripts/fcloud/fcloud_workflow.py`), 5 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#420-420), lines 420 to 450

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#680-680), lines 680 to 720

Searched for regex `add_parser|--quant-mode|--model_path|p_acc|p_full|p_quick` (`**/scripts/fcloud/fcloud_workflow.py`), 18 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#660-660), lines 660 to 682

Now I have enough to implement the fix. Apply edits:   

Replacing 69 lines with 106 lines in [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py)


```

```


Checked [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py), no problems found

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py accuracy --help 2>&1 | tail -10 && echo --- && python3 scripts/fcloud/fcloud_workflow.py full --help 2>&1 | tail -10

CLI works. Now create the documentation:   

Searched for files matching `**/docs/soar_2026_changes/CHANGE_013*`, 8 matches

Next number is 0134. Create EN+ZH docs:   

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add scripts/fcloud/fcloud_workflow.py docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.en.md docs/soar_2026_changes/CHANGE_0134_eval_model_path_quant_mode.zh.md && git commit -m "fix(fcloud): eval --model_path must match served model — CHANGE_0134

step_accuracy/step_quick_accuracy hardcoded --model_path to the GPTQ
path regardless of which model the server was actually serving. The
eval harness uses --model_path to load the tokenizer, chat template,
and GenerationConfig (eos/stop). When server runs the BF16 non-quanted
model but eval loads tokenizer from the GPTQ dir, prompts/stop tokens
may be wrong, causing runaway generation to max_tokens=65536 and 3000s
read timeouts (the symptom seen in Round 13e Test 1).

Fix: add _resolve_model_path(quant_mode, model_path) helper; thread
--quant-mode / --model-path into accuracy / quick-accuracy / full CLI
subcommands. Default --quant-mode=gptq preserves prior behaviour for
all existing dense-path tests.

Also documents the sparse-path activation rules (requires both
--attention-backend minicpm_flashinfer AND no --force-dense-minicpm;
--dense-as-sparse is not an activator). See CHANGE_0134 docs." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Pushed `ae119f5aa`. Ready for your fcloud restart.

## Summary

**Sparse activation rule (corrected)**: requires **both** `--attention-backend minicpm_flashinfer` (or `minicpm_flashattn`) **AND** absence of `--force-dense-minicpm`. `--dense-as-sparse` is NOT an activator — it only widens sparse compute coverage when sparse is already on. Therefore the official toolkit default (`--attention-backend flashinfer`) does NOT run the custom sparse backend; it's effectively dense for sparse layers.

**Fix shipped**:
- fcloud_workflow.py — `accuracy`, `quick-accuracy`, `full` subcommands now accept `--quant-mode {gptq,fp8_blockwise,noquant}` and `--model-path`. Eval `--model_path` is resolved per quant_mode so tokenizer/EOS/chat_template match what the server is actually serving. Default still `gptq` (no behavior change for dense submission tests).

**Docs**: `CHANGE_0134_eval_model_path_quant_mode.{en,zh}.md` (sparse-activation clarification + bug + fix + risks).

**When you start fcloud**, the recommended first action is to run the three `diff` commands listed in the doc to see if the GPTQ vs non-quanted model dirs have different tokenizer/chat_template/generation_config. The result decides whether the Round 13e timeouts were a tokenizer-mismatch artifact (worth re-testing) or genuine sparse slowness (closes the line). Awaiting your go-ahead on fcloud start. 



