你已经批准了 `CHANGE_0051`。我先读现有评测脚本并从公开集抽出固定 quick-screen 样本，然后一次性落脚本、数据集和文档。



Searched for files matching `**/docs/soar_2026_changes/CHANGE_0051*`, no matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#1-1), lines 1 to 260

Fetched 2 resources

我先抽出固定的 20 条 quick-screen 样本，并按你当前最弱桶来配比。然后我会落一个轻量脚本，直接给出 `mcq`、中长 `qa` 的快速代理分数。

Ran terminal command:  python3 - <<'PY'
import json
from pathlib import Path
path = Path('benchmark/soar/demo_sala/perf_public_set.jsonl')
rows = {obj['index']: obj for obj in map(json.loads, path.read_text(encoding='utf-8').splitlines()) if obj}
indices = [1, 2, 4, 8, 11, 17, 23, 25, 61, 63, 65, 66, 68, 70, 71, 76, 80, 81, 85, 90]
for idx in indices:
    print(json.dumps(rows[idx], ensure_ascii=False))
PY

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#261-261), lines 261 to 466

我已经拿到筛选集的精确行内容引用和现有评测实现。接下来直接落 quick-screen 脚本、固定 20 条数据集和 CHANGE_0051 文档，然后做静态校验。

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_NASDlNcWJCVm8wzwMR2hWEw6__vscode-1774137058717/content.txt)

Compacted conversation现在开始改三个点：固定 quick-screen 样本定义、轻量评测脚本、双语文档。实现会尽量复用现有评测逻辑，避免引入新的判分分叉。Optimizing tool selection...

Created 5 todos

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala)

Searched for files matching `sglang` · `docs/soar_2026_changes/CHANGE_005*.md`, 2 matches

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#1-1), lines 1 to 260

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py)

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/eval_model_001.py#261-261), lines 261 to 466

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0050_calibration_candidate_builder.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0050_calibration_candidate_builder.zh.md)

实现上我准备不复制一份 20 条 JSONL 原文，而是保存固定 subset 的索引配置并在脚本里从 `perf_public_set.jsonl` 读取。这样更稳，避免公共集内容重复散落两份。

Read memory

Read memory /memories/repo/

我要落地的编辑是：
1. 新增固定 20 条 subset 配置文件，避免复制公共集原文。
2. 新增 quick-screen 脚本，复用现有打分逻辑并输出 focus buckets 指标。
3. 新增 CHANGE_0051 中英双语文档，写清验证和回滚方式。

Generating patch (556 lines) in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_screen_public_subset.json), [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_calibration_screen.py), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0051_quick_calibration_screen.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0051_quick_calibration_screen.zh.md)


```

```

```

```

```

```

```

```
代码已经加上。现在做两类校验：静态语法校验，以及 `--check` 路径校验，确认 subset 配置和源公开集一致。

Ran terminal command:  python3 -m py_compile benchmark/soar/demo_sala/quick_calibration_screen.py && python3 benchmark/soar/demo_sala/quick_calibration_screen.py --check
Completed with input:  python3 -m py_compile benchmark/soar/demo_sala/quick_calibration_screen.py && python3 benchmark/soar/demo_sala/quick_calibration_screen.py --check

Checked [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_calibration_screen.py), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0051_quick_calibration_screen.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0051_quick_calibration_screen.zh.md), no problems found

`--check` 路径暴露出一个实际问题：现有 eval_model_001.py 顶层导入了不可用的 `Engine`，导致我的脚本在导入它时直接失败。我要把 quick-screen 需要的最小逻辑内联成自包含实现，绕开这个外部导入点。

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_calibration_screen.py#1-1), lines 1 to 220

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_calibration_screen.py#221-221), lines 221 to 387

Generating patch (272 lines) in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_calibration_screen.py)


```

```


Ran terminal command:  python3 -m py_compile benchmark/soar/demo_sala/quick_calibration_screen.py && python3 benchmark/soar/demo_sala/quick_calibration_screen.py --check

Checked [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/quick_calibration_screen.py), no problems found

静态和 `--check` 校验都通过了。额外发现并规避了一个仓库里的真实问题：eval_model_001.py 顶层依赖当前环境里不可导入的 `Engine`，所以 quick-screen 现在是自包含实现，不再卡在那个导入点。

Created memory file quick_screen_notes.md

Completed: *Inspect eval script details* (1/5)

新增了一个可直接用来筛 calibration 候选的 quick-screen 工具，核心是固定 20 条公开集样本做 bucket-aware 代理评测，避免每个候选都先跑 1 小时完整评测。

**变更**
- 新增 quick-screen 子集配置：quick_screen_public_subset.json
- 新增快速评测脚本：quick_calibration_screen.py
- 新增英文文档：CHANGE_0051_quick_calibration_screen.en.md
- 新增中文文档：CHANGE_0051_quick_calibration_screen.zh.md

**实现要点**
- 固定 subset 索引为 `1, 2, 4, 8, 11, 17, 23, 25, 61, 63, 65, 66, 68, 70, 71, 76, 80, 81, 85, 90`
- 不复制第二份大 JSONL，而是用索引配置从公开集读取，减少重复和漂移风险
- 输出三类核心指标：
  - `quick_screen_accuracy`
  - `focus_bucket_average`
  - `focus_bucket_min`
- focus buckets 固定为：
  - `task=mcq|len_0_4k`
  - `task=qa|len_4k_32k`
  - `task=qa|len_32k_128k`

**额外处理**
- 现有 eval_model_001.py 顶层导入了当前环境中不可用的 `Engine`，直接复用会让 quick-screen 连 `--check` 都起不来。
- 所以 quick-screen 脚本做成了自包含实现，保留相同的打分和 bucket 统计逻辑，但不再依赖那个导入点。

**验证**
已通过：
- `python3 -m py_compile benchmark/soar/demo_sala/quick_calibration_screen.py`
- `python3 quick_calibration_screen.py --check`

`--check` 结果确认：
- subset size = 20
- task counts = `{"mcq": 8, "qa": 12}`

**使用方式**
先做配置校验：
```bash
python3 benchmark/soar/demo_sala/quick_calibration_screen.py --check
```

再做实际 quick screen：
```bash
python3 benchmark/soar/demo_sala/quick_calibration_screen.py \
  --model_path openbmb/MiniCPM-SALA \
  --api_base http://127.0.0.1:30000
```

结果会写到：
- `outputs/quick_screen_<timestamp>/predictions.jsonl`
- `outputs/quick_screen_<timestamp>/summary.json`

这里没有实际跑推理请求，因为当前会话里没有验证可用的 SGLang 服务端点；只完成了静态和 subset 配置校验。

1. 如果你愿意，我下一步可以直接给你一套 quick-screen 优先测试顺序，先筛 `candidate_baseline_like_8` / `candidate_mcq_qa_mid_8` / `candidate_mcq_heavy_8` / `candidate_qa_balanced_8`。
2. 你也可以先在 fcloud 上跑一轮 quick-screen，把 `summary.json` 发我，我再帮你决定哪些候选值得进 1 小时完整评测。

Made changes.
