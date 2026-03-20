I’m reviewing the copied eval script against the local suite and the dataset format, focusing on how it computes ori_accuracy and whether anything in this folder setup could mis-score perf_public_set.jsonl.

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model.py#1-1), lines 1 to 260

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model.py#261-261), lines 261 to 443

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/perf_public_set.jsonl#1-1), lines 1 to 12

I found a likely scoring inflation path in long-context rows and a separate prompt-handling mismatch in generation. I’m pulling exact line numbers for those before I summarize the review.

Searched for regex `def score_exact_match|task in \['qa', 'niah', 'lcx'\]|hits = any|hits = sum` (`**/benchmark/soar/demo_sala/eval_model.py`), 4 results

Searched for regex `messages = \[self.tokenizer.apply_chat_template|raw_inputs = inputs|executor.submit\(_infer, i, raw_inputs\[i\]\)|mode='mid'|max_out_len=65536` (`**/benchmark/soar/demo_sala/eval_model.py`), 5 results

Searched for regex `"task": "niah".*"gold": \[` (`**/benchmark/soar/demo_sala/perf_public_set.jsonl`), 20 results

I have the main review points. I’m grabbing one last exact reference for the output-token accounting, because it affects the reported totals and can mislead performance interpretation even though it does not change ori_accuracy.

Searched for regex `def get_token_len|out_len = model.get_token_len\(pred\)|total_output_tokens` (`**/benchmark/soar/demo_sala/eval_model.py`), 11 results

Searched for regex `Original Accuracy|Normalized Accuracy|ori_accuracy|overall_accuracy` (`**/benchmark/soar/demo_sala/eval_model.py`), 4 results

**Findings**

1. High: Long-context rows with multiple required answers are over-scored as fully correct on any single hit.  
The scoring path treats niah, qa, and lcx as full-credit if any gold string appears in the prediction, in eval_model.py and eval_model.py. But perf_public_set.jsonl contains niah rows that explicitly ask for all answers and provide multiple gold items, for example perf_public_set.jsonl, perf_public_set.jsonl, perf_public_set.jsonl, and perf_public_set.jsonl.  
Impact: a model can output 1 of 4 required values and still receive 1.0 for that sample, which inflates ori_accuracy materially.

2. Medium: The prompt-formatting and truncation path is internally inconsistent and mostly bypassed.  
The code builds chat-templated messages in eval_model.py, but then discards them and sends raw_inputs instead in eval_model.py and eval_model.py. At the same time, the runner instantiates the model in mid mode in eval_model.py, but the raw-input mid-truncation branch is commented out.  
Impact: local eval is not actually exercising the templated/truncated prompt path the wrapper appears to intend, so long-context behavior can differ from what the code suggests.

3. Medium: The reported output token count and TPS are not trustworthy.  
Output length is measured with get_token_len(pred) in eval_model.py, but get_token_len applies the chat template to its argument as if it were a user prompt in eval_model.py.  
Impact: total_output_tokens, TPS, and the summary total_tokens field can be significantly distorted. This does not change ori_accuracy directly, but it can mislead any speed interpretation based on this script’s summary.

4. Medium: The eval run is configured to allow extremely long generations for every sample.  
The generation call uses max_out_len=65536 in eval_model.py. For MCQ rows in perf_public_set, that is far above what is needed.  
Impact: this can increase runtime variance, encourage overlong reasoning, and make local durations look worse than necessary. It also increases the chance that answer extraction succeeds or fails for reasons unrelated to the underlying model quality.

**Open Questions**

- Does the official toolkit eval_model.py also intentionally score multi-answer niah rows as any-hit full credit, or was this copied script locally modified for convenience? That is the main thing I would verify before trusting the reported ori_accuracy.
- Is the intent to emulate the OpenAI chat API path exactly, or to evaluate the model with an explicit local chat template? The current script mixes both approaches.

**What looks fine**

- The copied script is structurally compatible with your copied dataset. It expects question, task, and gold, and perf_public_set.jsonl has those fields, for example perf_public_set.jsonl.
- run_soar_suite.py should be able to consume this script’s stdout, because it prints Average Score and writes ori_accuracy/overall_accuracy in its summary JSON, and the wrapper already regex-parses Average Score and ori_accuracy from eval output.

**Change Summary**

- No code changes made.
- The main review conclusion is that this eval_model.py is usable for local smoke testing, but I would not trust its ori_accuracy as a faithful long-context score until the multi-answer niah scoring issue is resolved.

1. If you want, I can next give you a minimal patch proposal to make niah scoring require all listed answers while leaving the rest of the script unchanged.
2. If you prefer, I can compare this copied eval_model.py against the original toolkit version line by line and isolate exactly what was changed.
