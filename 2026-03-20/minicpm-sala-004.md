是的，基本上是这样。

如果你想复现“以前那种顺序取前几个样本”的量化行为，那么设置：

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential
export SOAR_GPTQ_CALIBRATION_SAMPLES=8
```

或者：

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential
export SOAR_GPTQ_CALIBRATION_SAMPLES=16
```

会把“校准样本选择方式”恢复成旧行为，也就是按 preprocess_model.py 的顺序前 N 条。

但要注意，这只能复现“采样行为像以前”，不保证整个量化结果和你记忆中的那次完全一样。还需要这些条件也一致：

1. `SOAR_GPTQ_GROUP_SIZE` 一样
2. `SOAR_GPTQ_ATTN_IMPL` 一样
3. `SOAR_GPTQ_INCLUDE_MODULES` / `SOAR_GPTQ_EXCLUDE_MODULES` 一样
4. `SOAR_GPTQ_LAYER_AWARE` 一样
5. `gptqmodel` 版本一样
6. perf_public_set.jsonl 内容一样
7. `calibration_field` 仍然是 `question`

如果这些都没变，那么：
- `sequential + 8`
- `sequential + 16`

就是你最接近“以前量化方式”的回放。

另外，`sequential` 模式下 seed 不起作用，所以：
- `SOAR_GPTQ_CALIBRATION_SEED` 无关紧要

对于你第二个想法，我认为是合理的，而且很有用。

不是为了正式提交分数，而是为了诊断“到底是哪类题在掉点”。  
如果我们能把正确率拆成：

1. 按 `task` 的准确率
2. 按 `task + prompt_tokens length bucket` 的准确率

那就能回答这种问题：

- 是不是 `mcq` 在掉，而长上下文没掉
- 是不是 `niah` 的 `32k-128k` 桶掉得最厉害
- 是不是 `qa` 在超长输入上特别脆弱

这正好能反过来指导 calibration sampling 的配额设计，所以方向是对的。

但按仓库要求，我这里先给你一个修改提案，不直接动代码，等你批准后我再改。

**提案**

目标：
- 复制 eval_model.py 为新的诊断脚本 `eval_model_01.py`
- 在保持原有 `ori_accuracy` / `overall_accuracy` 输出兼容的前提下，新增分任务和分长度桶的准确率统计

预期收益：
1. 不改变现有 correctness 主分数计算方式
2. 额外得到 task-level / task+length-level 的诊断信息
3. 便于后续决定 calibration 样本应向哪个类别倾斜

规则合规检查：
1. 这是本地诊断脚本，不改官方评测逻辑
2. 不替换模型，不改服务参数，不碰官方并发约束
3. 只用于自查分析，合规

风险：
1. 这是诊断增强，不应影响正式分数
2. 如果不小心改动原 scoring 路径，可能导致和官方脚本不一致
3. 所以我会保持原有总分逻辑不变，只增加额外统计

拟改文件：
1. 新增 benchmark/soar/demo_sala/eval_model_01.py

拟改内容：
1. 复制当前 eval_model.py
2. 保留现有：
   - `Average Score`
   - `ori_accuracy`
   - `overall_accuracy`
   - `summary.json`
3. 新增：
   - 每个 `task` 的 `count / score / accuracy`
   - 每个 `task + prompt_tokens bucket` 的 `count / score / accuracy`
4. 新增长度桶，建议和 calibration 那边保持一致：
   - `0-4k`
   - `4k-32k`
   - `32k-128k`
   - `128k+`
5. 把这些统计同时：
   - 打印到 stdout
   - 写入新的 JSON 文件，例如 `breakdown.json`

验证命令：
```bash
python3 benchmark/soar/demo_sala/eval_model_01.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

成功标准：
1. 原有 `Average Score` 不变
2. 新脚本能输出 task 维度统计
3. 新脚本能输出 task+length bucket 维度统计
4. 结果能直接用于指导 calibration 样本配额

失败标准：
1. 总分和原脚本不一致
2. breakdown 统计和样本数对不上
3. 输出格式太乱，不便于你在 fcloud 日志里看

如果你批准，我下一步就做这一项：
- 新增 `eval_model_01.py`
- 保持原分数逻辑不变
- 只加诊断统计和输出文件
