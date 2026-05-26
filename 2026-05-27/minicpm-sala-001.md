Searched for text `MODEL_PATH` (`**/scripts/fcloud/fcloud_workflow.py`), 20 results

Searched for regex `add_parser|restart-server|argparse|args\.command` (`**/scripts/fcloud/fcloud_workflow.py`), 20 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#685-685), lines 685 to 760

Good news — option B is **already implemented**. `restart-server`, `accuracy`, `quick-accuracy`, and `full` all accept `--model-path` and `restart-server` accepts `--env KEY=VAL` (see fcloud_workflow.py). No code change needed.

## Concrete plan for the SAMPLES=30 sequential test

After your approval to start fcloud, the sequence is:

```bash
# 1) Resume instance
python3 scripts/fcloud/fcloud_workflow.py start-instance

# 2) Re-quantize into NEW folder (canonical 90-stratified ckpt untouched).
#    Run on fcloud directly — the workflow has no quantize subcommand.
python3 scripts/fcloud/fcloud_exec.py exec '
cd /root/submission_sim &&
source prepare_env.sh &&
export SOAR_GPTQ_CALIBRATION_SAMPLES=30 &&
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential &&
bash prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/openbmb/MiniCPM-SALA-30-sequential-sparse_qkv_w8
' --timeout 5400

# 3) Restart server pointing at the new ckpt + matching calib env
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/openbmb/MiniCPM-SALA-30-sequential-sparse_qkv_w8 \
  --env SOAR_GPTQ_CALIBRATION_SAMPLES=30 \
  --env SOAR_GPTQ_CALIBRATION_SAMPLING=sequential

python3 scripts/fcloud/fcloud_workflow.py wait-server

# 4) Eval & bench (point eval at the same ckpt so its tokenizer/chat_template match)
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --model-path /root/models/openbmb/MiniCPM-SALA-30-sequential-sparse_qkv_w8

python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# 5) Pause
python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

A couple of notes:
- The `accuracy` step's `--model-path` only changes which tokenizer/template the eval harness loads; the served model is whichever the server points at (step 3). I'm passing the same path to keep them in sync.
- After results are in, I'll append a `GPTQ-FP8-DENSE-S30-seq` row to TEST_RESULTS_TRACKING + bilingual chat log + commit/push.
- The 90-stratified ckpt at `…-90-qa-cwe-mcq-sparse_qkv_w8/` stays intact for any future A/B.

**Awaiting your explicit "go" before touching fcloud.**
