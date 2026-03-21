我先把这次变更的范围收紧成一个独立 feature：新增两份“语义优先”的校准子集文件，并同步补一对 CHANGE_0047 文档，然后做内容校验。

Created 3 todos

我先从公开集里提取这两组 index 对应的原始 JSONL 记录，确认文件内容精确无误，再一次性落到 demo_sala 目录。

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
import json
from pathlib import Path
path = Path('benchmark/soar/demo_sala/perf_public_set.jsonl')
rows = [json.loads(line) for line in path.open() if line.strip()]
row_by_index = {row['index']: row for row in rows}
indices = [2, 11, 17, 23, 25, 61, 76, 90]
for idx in indices:
    print(json.dumps(row_by_index[idx], ensure_ascii=False))
PY

Ran terminal command:  cd /home/oldzhu/sglang && python3 - <<'PY'
import json
from pathlib import Path
path = Path('benchmark/soar/demo_sala/perf_public_set.jsonl')
rows = [json.loads(line) for line in path.open() if line.strip()]
row_by_index = {row['index']: row for row in rows}
indices = [2, 5, 8, 11, 17, 23, 25, 30, 61, 63, 66, 76, 81, 90, 31, 60]
for idx in indices:
    print(json.dumps(row_by_index[idx], ensure_ascii=False))
PY

提取结果已经写到临时文件里。我现在读回这两份精确 JSONL 内容，然后直接生成校准子集文件和文档。

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_Mip9hJKkZ0DaSd935X4A9I8v__vscode-1774050053614/content.txt#1-1), lines 1 to 8

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/call_3XmTexNTjrjHZH8U1XAiAwbs__vscode-1774050053615/content.txt)

Compacted conversation优化提案如下。这个仓库要求改动前先提案并等你明确批准，所以我先把要做的内容定清楚。

**目标**
- 在 demo_sala 下新增两份“语义优先”GPTQ 校准子集文件，便于你在 fcloud 直接做 A/B：
  - 8 条子集：索引 `2, 11, 17, 23, 25, 61, 76, 90`
  - 16 条子集：索引 `2, 5, 8, 11, 17, 23, 25, 30, 31, 60, 61, 63, 66, 76, 81, 90`
- 同时提供一个可复现的生成脚本/命令，避免手工维护 JSONL。
- 按仓库要求补一组中英文变更文档，作为一个独立 feature 迭代。

**预期收益**
- 让当前 GPTQ 路线先验证“样本组成”这个最可疑变量，而不是继续混入 kernel / runtime 干扰项。
- 比当前 `sequential` / `stratified` 更聚焦 `mcq + qa`，只少量保留 `niah`，避免 `cwe/fwe` 这类低语义样本主导校准。

**规则合规检查**
- 只使用官方公开数据 perf_public_set.jsonl 的子集，不引入外部数据。
- 不改推理逻辑，不改评测协议，不触碰任何 SOAR 禁止项。
- 这属于离线量化校准数据选择，符合官方技术路径里允许的模型预处理优化方向。

**风险**
- 这是低风险改动，因为只新增校准输入文件和生成脚本，不动运行时代码。
- 主要风险是这两组子集不一定优于你现有的 `sequential 8`，但验证成本低，且结果能直接指导下一步是否做“快速精度筛选工具”。

**准备改动的文件**
- 新增校准子集文件：
  - benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
  - benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
- 新增一个小生成脚本，固定索引并可重建上述文件：
  - benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
- 新增文档：
  - `docs/soar_2026_changes/CHANGE_0047_semantic_priority_calibration_subsets.en.md`
  - `docs/soar_2026_changes/CHANGE_0047_semantic_priority_calibration_subsets.zh.md`

**验证命令**
- 文件格式校验：
```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
```
- 行数校验：
```bash
wc -l benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
wc -l benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
```
- 量化测试建议：
```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```
```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

**成功/失败判据**
- 成功：在接近当前速度前提下，`ori_accuracy` 明显高于你现有的通用 `sequential 8/16`。
- 失败：精度没有改善，或波动仍然很大；那就该进入下一步，做一个 lightweight 的量化前快速精度筛选工具。

如果你批准，我下一步就直接创建这 5 个文件并做本地校验。请回复 `批准`。
