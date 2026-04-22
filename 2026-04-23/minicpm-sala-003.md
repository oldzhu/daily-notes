Good question. Let me investigate what quick verification options exist in the eval pipeline. 

Research fast accuracy verification methods

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#1-1), lines 1 to 200

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model.py#1-1), lines 1 to 200

Ran terminal command:  wc -l /home/oldzhu/sglang/benchmark/soar/demo_sala/perf_public_set.jsonl && head -1 /home/oldzhu/sglang/benchmark/soar/demo_sala/perf_public_set.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#200-200), lines 200 to 400

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model.py#200-200), lines 200 to 400

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import json
from collections import Counter
tasks = Counter()
with open('benchmark/soar/demo_sala/perf_public_set.jsonl') as f:
    for line in f:
        d = json.loads(line)
        tasks[d['task']] += 1
for task, count in sorted(tasks.items()):
    print(f'{task}: {count}')
print(f'TOTAL: {sum(tasks.values())}')
"

Searched for text `accuracy` (`**/scripts/fcloud/fcloud_workflow.py`), 17 results

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import json
with open('benchmark/soar/demo_sala/perf_public_set.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        print(f'{i}: task={d["task"]} prompt_tokens={d["prompt_tokens"]} completion_tokens={d["completion_tokens"]} gold_type={type(d["gold"]).__name__} gold_preview={str(d["gold"])[:80]}')
        if i >= 9: break
"

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#200-200), lines 200 to 260

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import json
# Show token distribution per task
from collections import defaultdict
stats = defaultdict(lambda: {'count':0, 'prompt_min':999999, 'prompt_max':0, 'prompt_sum':0, 'comp_sum':0})
with open('benchmark/soar/demo_sala/perf_public_set.jsonl') as f:
    for line in f:
        d = json.loads(line)
        t = d['task']
        s = stats[t]
        s['count'] += 1
        s['prompt_sum'] += d['prompt_tokens']
        s['comp_sum'] += d['completion_tokens']
        s['prompt_min'] = min(s['prompt_min'], d['prompt_tokens'])
        s['prompt_max'] = max(s['prompt_max'], d['prompt_tokens'])
for task in sorted(stats):
    s = stats[task]
    print(f'{task}: n={s["count"]} prompt_tokens=[{s["prompt_min"]}..{s["prompt_max"]}] avg_prompt={s["prompt_sum"]//s["count"]} avg_comp={s["comp_sum"]//s["count"]}')
"

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#500-500), lines 500 to 558

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#380-380), lines 380 to 466

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model.py#380-380), lines 380 to 443

Great news — `--num_samples` is already supported. The eval file is ordered MCQ-first (30 samples), so `--num_samples 30` gives an MCQ-only test in ~3-5 minutes. But MCQ alone won't catch regressions in qa/cwe/niah.

Here's the plan:

**Fast verification approach**: Add a `quick-accuracy` mode to the workflow that runs a **stratified subset** — a few samples from each task type. This gives ~8-10 minute turnaround instead of 50-60 minutes.

Let me implement this and then set up the Phase B testing matrix.  

Created 3 todos

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#140-140), lines 140 to 220

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#220-220), lines 220 to 330

I see — `--num_samples` just takes the first N samples in file order. Let me check the file ordering:   

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import json
with open('benchmark/soar/demo_sala/perf_public_set.jsonl') as f:
    tasks = [json.loads(l)['task'] for l in f if l.strip()]
# Show task order
from collections import Counter
seen = []
for i, t in enumerate(tasks):
    if not seen or seen[-1][0] != t:
        seen.append((t, i, 1))
    else:
        seen[-1] = (t, seen[-1][1], seen[-1][2]+1)
for t, start, count in seen:
    print(f'  {t}: samples {start}-{start+count-1} ({count} samples)')
print(f'Total: {len(tasks)}')
"

Now let me create a **stratified quick-eval subset** and add support to the workflow:  

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
import json
with open('benchmark/soar/demo_sala/perf_public_set.jsonl') as f:
    samples = [json.loads(l) for l in f if l.strip()]

# Show prompt token lengths per task (approximate by character count / 3.5)
from collections import defaultdict
by_task = defaultdict(list)
for i, s in enumerate(samples):
    by_task[s['task']].append((i, len(s.get('question','')) or len(str(s))))
    
for task in ['mcq','niah','qa','fwe','cwe']:
    items = by_task[task]
    items.sort(key=lambda x: x[1])
    print(f'{task}: shortest 5 indices: {[(idx, length) for idx, length in items[:5]]}')"

MCQ is tiny (~400 chars) while long-context tasks are 37K-128K chars. A stratified quick test with all 30 MCQ + 2 shortest per long-context task would be ~36 samples but still take a while due to long prefills.

Better approach: add `--task-filter` and `--num-samples-per-task` to the eval script so we can do targeted checks. Let me implement this:  

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#300-300), lines 300 to 340

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#160-160), lines 160 to 170

Now let me add `--task-filter` and `--num-samples-per-task` to the eval script, and a `quick-accuracy` command to the workflow:   

Replacing 2 lines with 4 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py)


```

```
Now modify the dataset loading to support these filters:    

Replacing 10 lines with 28 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py)


```

```
Now add the `quick-accuracy` command to fcloud_workflow.py:     

Searched for regex `def step_accuracy|def main|subcommands|subparsers|add_parser.*accuracy` (`**/scripts/fcloud/fcloud_workflow.py`), 5 results

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#200-200), lines 200 to 240

Read [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py#510-510), lines 510 to 558

Replacing 7 lines with 43 lines in [](file:///home/oldzhu/sglang/scripts/fcloud/fcloud_workflow.py)


```

```
Now add GPTQModel calibration tuning parameters (`damp_percent`, `mse`) to preprocess_model.py:      

Starting: *Add damp_percent/mse params to preprocess* (2/3)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#885-885), lines 885 to 895

I need to check what QuantizeConfig actually accepts to make sure we pass the right params: 

Ran terminal command: from gptqmodel import QuantizeConfig; import inspect; print(inspect.signature(QuantizeConfig.__init__))" 2>/dev/null || echo "GPTQModel not installed locally

Ran terminal command:  cd /home/oldzhu/sglang && pip show gptqmodel 2>/dev/null | head -5; find / -name "quantize_config.py" -path "*/gptqmodel/*" 2>/dev/null | head -3

GPTQModel isn't installed locally. Let me check the whl file on fcloud or look at the GPTQModel API docs:  



Fetched https://raw.githubusercontent.com/ModelCloud/GPTQModel/main/gptqmodel/quantization/config.py

Now I have the full picture. `GPTQConfig` supports `damp_percent`, `mse`, and `foem` parameters. Let me add env var controls for these in preprocess_model.py: 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#888-888), lines 888 to 895

Searched for regex `def _parse_float_env|def _parse_int_env` (`**/benchmark/soar/demo_sala/preprocess_model.py`), 1 result

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#84-84), lines 84 to 96

I need a float parser. Let me add one and update the QuantizeConfig call:    

Replacing 11 lines with 21 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py)


```

```
Now add the calibration tuning parameters to the QuantizeConfig construction:     

Replacing 1 lines with 7 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py)


```

```
Now update the logging line to include the new params:      

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#898-898), lines 898 to 910

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#910-910), lines 910 to 925

Replacing 9 lines with 10 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py)


```

```
Now also add the env vars to `prepare_env.sh`:      

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#110-110), lines 110 to 125

Replacing 3 lines with 5 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)


```

```
Compacted conversation

Starting: *Propose Phase B test matrix* (3/3)

Now let me verify all changes are consistent and propose the Phase B test plan. Let me first verify the changes compile correctly: 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#84-84), lines 84 to 110

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#898-898), lines 898 to 925

Good. All changes look correct. Now let me propose the Phase B test plan:

---

## Phase B: Dense Calibration Tuning — Optimization Proposal

**Objective**: Recover accuracy lost with dense calibration (Test 24: 77.64%) while keeping its speed benefit (~1.5-2.7% faster than sparse calibration baseline).

**Changes implemented** (pending commit/push):
1. **`_parse_float_env()`** helper added to preprocess_model.py
2. **`SOAR_GPTQ_DAMP_PERCENT`** env var (default 0.05) → controls Hessian damping
3. **`SOAR_GPTQ_MSE`** env var (default 0.0) → enables MSE grid search when >0
4. Both added to prepare_env.sh for submission compatibility
5. eval_model_001.py — `--task_filter` + `--num_samples_per_task` for fast screening
6. fcloud_workflow.py — `quick-accuracy` subcommand for MCQ-only fast tests

**Phase B Test Matrix** (each requires ~85 min re-quantization + ~3-5 min quick-accuracy screening):

| Test | Config | Env Vars | Rationale |
|------|--------|----------|-----------|
| B1 | Dense + higher damping | `SOAR_GPTQ_DAMP_PERCENT=0.1` | More regularization → more stable Hessian inverse |
| B2 | Dense + much higher damping | `SOAR_GPTQ_DAMP_PERCENT=0.2` | Aggressive regularization |
| B3 | Dense + MSE grid search | `SOAR_GPTQ_MSE=1.0` | MSE-optimized weight rounding |
| B4 | Dense + damping + MSE | `SOAR_GPTQ_DAMP_PERCENT=0.1 SOAR_GPTQ_MSE=1.0` | Combined |

**Screening protocol**:
1. Re-quantize with each config (~85 min)
2. Quick-accuracy (MCQ-only, 30 samples, ~3-5 min) — if MCQ < 50%, skip
3. Full accuracy only for configs with MCQ ≥ 55%
4. Speed test only for configs with ori_accuracy ≥ 79%

**Total time estimate per test**: ~90 min (quantize + quick-accuracy). Full eval adds ~50-60 min if needed.

**Next steps**:
1. Commit and push all changes to `minicpm-src`
2. User starts fcloud instance
3. Sync + run B1 first (safest — just increased regularization)

Shall I commit and push these changes now? 

Completed: *Propose Phase B test matrix* (3/5)



