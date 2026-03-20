我先复核 SOAR 官方规则页和 toolkit 页里的最新说明，确认这个“校准采样策略优化”是否合规，然后给你一版可直接审批的变更提案，不做代码修改。



Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1-1), lines 1 to 220

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)

Fetched 2 resources

**提案**

目标：把 GPTQ 校准从“顺序取前 N 条 question”升级为“可控采样的公开集校准”，优先改善 4bit 量化后的正确率稳定性，尤其是长上下文任务退化。

预期收益：
1. 提高校准样本对真实公开/私有同分布任务的覆盖度，减少“前 64 条样本偏置”。
2. 在不改变推理路径的前提下，争取把 ori_accuracy 拉回更稳定区间。
3. 相比继续单纯增大 SOAR_GPTQ_CALIBRATION_SAMPLES，这条路径更可能带来实质收益。

**规则合规检查**

我刚复核了官方 competition 和 toolkit 页面，结论是这条优化是合规的：

1. 官方明确公开了 perf_public_set.jsonl 对应的数据集与 eval_model.py 供选手自查。
2. 官方技术路径指引明确推荐“路径一：量化加速”，并允许在评测机现场执行 GPTQ 量化流程。
3. 当前方案只改变公开集上的离线校准采样策略，不引入私有集信息，不修改官方 correctness 评测逻辑，不触碰 prefix cache 或固定并发规则。
4. 这符合“可复现、可解释、在官方环境稳定运行”的要求。

**为什么我建议先做这个，而不是继续试别的参数**

当前代码里最明显的问题不是样本数，而是样本选取方式：

1. preprocess_model.py 到 preprocess_model.py 的 load_calibration_texts 只是顺序读前 N 条。
2. 当前默认只读 `question` 字段，这点我认为是对的，应该保持。
3. perf_public_set.jsonl 存在任务类型和长度分布差异，顺序前 64 条未必代表整体公开集，更不一定代表私有集同分布。

所以优先级我建议是：
1. 先改采样策略
2. 再评估 group_size=64
3. 最后才考虑 attention impl 或更激进模块覆盖

**拟议改动**

本次只做一个 cohesive feature：公开集校准样本选择策略。

改动文件：
1. preprocess_model.py
2. prepare_env.sh
3. 审批后新增双语文档：
   - `docs/soar_2026_changes/CHANGE_0046_...en.md`
   - `docs/soar_2026_changes/CHANGE_0046_...zh.md`

拟改函数和位置：
1. preprocess_model.py
   - 扩展 `load_calibration_texts(...)`
   - 从“只返回前 N 条文本”改为“先读取记录，再按策略选样本，再取 text_field”
2. preprocess_model.py
   - `run_gptq_quantization(...)` 增加采样策略相关参数打印，便于日志确认实际生效策略
3. preprocess_model.py
   - argparse 增加可选采样控制参数，或从环境变量读取
4. prepare_env.sh
   - 增加新的 `SOAR_GPTQ_*` 环境变量默认值和日志打印

**具体方案**

我建议实现三个可选模式，默认保持向后兼容：

1. `sequential`
   - 当前行为
   - 顺序取前 N 条
   - 作为回滚和基线

2. `shuffled`
   - 对公开集记录做固定种子打乱后取前 N 条
   - 低风险，比 sequential 更合理
   - 适合作为第一步替代默认策略

3. `stratified`
   - 按 task 和 prompt_tokens 分层取样
   - 这是我最看好的模式
   - 优先保证长上下文与短上下文都被覆盖

我建议的初始 stratified 逻辑不要过度复杂，避免引入不可控行为：

- 如果记录里有 `prompt_tokens`：
  - 短：`<= 4k`
  - 中：`4k-32k`
  - 长：`32k-128k`
  - 超长：`> 128k`
- 再结合 task：
  - `mcq`
  - `niah`
  - 其他

选择策略：
1. 先给每个有效 bucket 至少分 1 个名额
2. 其余名额按 bucket 中样本数比例分配
3. 不足时回填到剩余样本最多的 bucket
4. 使用固定 seed 保证可复现

这样做的好处：
1. 不需要你手工维护一个新的 calibration jsonl
2. 评测机上现场量化仍然完全可复现
3. 对隐藏私有集更稳，因为它本来就与公开集同分布

**建议新增环境变量**

放在 prepare_env.sh：

1. `SOAR_GPTQ_CALIBRATION_SAMPLING`
   - 可选：`sequential|shuffled|stratified`
   - 建议默认先设为 `shuffled` 或继续保持 `sequential` 以便保守上线

2. `SOAR_GPTQ_CALIBRATION_SEED`
   - 固定随机种子
   - 建议默认 `20260320`

3. `SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS`
   - `1|0`
   - 控制 stratified 是否使用 `prompt_tokens`

4. `SOAR_GPTQ_CALIBRATION_TASK_BALANCE`
   - `1|0`
   - 控制是否按 `task` 分层

我不建议第一版就加很多复杂配额参数。先把“可复现随机化 + 轻量分层”做干净。

**不建议在这次 feature 里做的事**

1. 不改 bits
2. 不改 include/exclude modules
3. 不把 `gold` 拼进校准文本
4. 不同时引入 group_size=64
5. 不改 serving 参数

原因很简单：这次 feature 的目标是隔离“校准样本选择”这个单变量，避免实验面失控。

**风险评估**

主要风险：
1. 公开集正确率不升反降
2. 长上下文提升但短题略退
3. 采样策略过复杂，导致复现和日志解释困难

风险等级：低到中

为什么我认为风险可控：
1. 默认保留 `sequential` 回滚路径
2. 不改推理链路，不动 kernel，不动服务参数
3. 只影响离线量化校准样本选择
4. 所有策略都基于公开集本身，不引入额外外部数据

**验证方案**

正确性验证：
1. 使用官方公开脚本跑完整公开集
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path ./perf_public_set.jsonl \
  --concurrency 32
```

2. 或跑本地封装
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path <MODEL_DIR> \
  --eval-script benchmark/soar/demo_sala/eval_model.py \
  --public-data benchmark/soar/demo_sala/perf_public_set.jsonl
```

量化验证：
```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```

建议做的 A/B：
1. `sequential + 32`
2. `sequential + 64`
3. `shuffled + 64`
4. `stratified + 64`

速度验证：
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path <MODEL_DIR> \
  --eval-script benchmark/soar/demo_sala/eval_model.py \
  --public-data benchmark/soar/demo_sala/perf_public_set.jsonl \
  --speed-data-s1 <S1数据集> \
  --speed-data-s8 <S8数据集> \
  --speed-data-smax <SMAX数据集>
```

**成功标准**

1. ori_accuracy 相比当前顺序采样基线有稳定改善，至少不再出现明显退化
2. 多次重复评测波动变小
3. 速度不因校准策略变化而明显恶化
4. 量化日志能明确打印出采样模式、seed、样本数、任务分布摘要

**失败标准**

1. ori_accuracy 持续不升反降
2. 不同重复 run 波动更大
3. 采样逻辑过于复杂，难以解释或复现
4. 官方环境现场量化出现额外不稳定行为

**回滚方案**

完全回滚很简单：
1. `SOAR_GPTQ_CALIBRATION_SAMPLING=sequential`
2. 恢复现有默认环境变量
3. 不需要改服务端或模型加载逻辑

**我的建议结论**

如果只做一个低风险、高信息量的 feature，我建议就是这个：
- 不再继续盲目增加 sample 数
- 只优化“从 perf_public_set.jsonl 里选哪 64 条 question 做校准”

这符合官方路径一，也符合你现在最需要解决的问题。

如果你批准，我下一步会按这个提案实现一个单 feature：
- `sequential/shuffled/stratified` 三种采样模式
- 固定 seed
- 日志打印采样摘要
- 同步补齐 CHANGE_0046 的中英文文档

回复我一个明确的批准即可，我再开始改代码。
