**Proposal**

Objective and expected gain

Implement one focused calibration feature: restrict GPTQ calibration sampling to `qa` and `cwe` rows only, while keeping 32 total samples and preserving length diversity across `len_4k_32k` and `len_32k_128k`.

Expected gain:
- recover quantization accuracy specifically in the buckets with the largest observed loss:
  - `task=qa|len_32k_128k`
  - `task=qa|len_4k_32k`
  - `task=cwe|len_4k_32k`
  - `task=cwe|len_32k_128k`
- keep speed essentially unchanged, because this only changes preprocessing calibration selection
- determine whether calibration focus alone is enough to get back above the 97% gate before moving to selective de-quantization

Rule-compliance check

I am treating this as compliant with the latest SOAR competition/toolkit guidance already reviewed in this workspace:

- still uses official on-platform preprocessing via prepare_model.sh
- does not replace the MiniCPM-SALA base model
- does not modify runtime concurrency, prefix cache, or evaluation logic
- only changes how the public calibration file is sampled during GPTQ preprocessing

Risk to accuracy/stability

Main upside:
- directly targets the observed quantization regression buckets instead of spending another iteration on broad sampling

Main risks:
- `mcq|len_0_4k` may regress if calibration becomes too narrow
- if the accuracy loss is driven more by quantization scope than by calibration composition, this may not recover enough
- overfitting to `qa` and `cwe` could hurt another bucket even if local overall score rises slightly

Why this is still worth trying:
- you only need a modest recovery to cross the 97% threshold band
- this is much lower-risk than changing kernel/runtime code again
- it is cleaner than immediately moving to selective de-quantization

Exact files/functions to change

Primary code change:
- preprocess_model.py

Likely functions involved:
- `_select_calibration_records(...)`
- `_calibration_bucket_key(...)`
- possibly `_parse_csv_env(...)` reuse for a new task include filter

Supporting config/env change:
- prepare_env.sh

Documentation required for this feature:
- `docs/soar_2026_changes/CHANGE_0059_gptq_qa_cwe_calibration_focus.en.md`
- `docs/soar_2026_changes/CHANGE_0059_gptq_qa_cwe_calibration_focus.zh.md`

Proposed implementation shape

Add one new environment variable:

- `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe`

Behavior:
- if set, calibration candidates are filtered to those tasks before sampling
- existing sample count logic still applies
- existing stratified logic still applies within the filtered set
- keep `SOAR_GPTQ_CALIBRATION_SAMPLES=32`
- keep `SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS=1`
- keep task/length-aware selection so both medium and long QA/CWE examples remain represented

Recommended initial submission config for this experiment:
- `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe`
- `SOAR_GPTQ_CALIBRATION_SAMPLES=32`
- `SOAR_GPTQ_CALIBRATION_SAMPLING=stratified`

Why I prefer this shape:
- smallest code change
- reversible by env only
- comparable against your current `32 stratified` baseline
- avoids creating a separate calibration file manually

Validation commands

Local preprocess check:
```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Local correctness eval:
```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <OUTPUT_MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

Recommended bucket comparison after local eval:
- compare against your current quant baseline on:
  - `task=qa|len_32k_128k`
  - `task=qa|len_4k_32k`
  - `task=cwe|len_32k_128k`
  - `task=cwe|len_4k_32k`

Success criteria

Success looks like:
- local `acc_ori` rises materially versus the current 32-sample stratified baseline
- especially in QA/CWE medium/long buckets
- no large speed change
- strong enough local signal to justify one official submission

Failure looks like:
- QA/CWE does not improve enough
- MCQ or another bucket collapses enough to offset the gain
- overall local accuracy still stays clearly below the needed threshold band

If it fails, the next feature should be:
- selective de-quantization / narrower GPTQ module scope for sensitive layers

Recommended next step

I recommend this as the next one-feature iteration before any more runtime/kernel work.

If you approve, I will implement:
1. task-filtered calibration sampling for `qa,cwe`
2. env wiring in prepare_env.sh
3. the required EN/ZH `CHANGE_0059` docs
4. syntax/diagnostic validation only, without changing anything unrelated
