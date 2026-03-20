我先把当前量化校准路径里的可调项重新过一遍，目标是只基于现有代码给出低风险、可能提升正确率的校准策略，不碰实现细节假设。

Searched for regex `SOAR_GPTQ_|include_modules|exclude_modules|attn_implementation|layer_aware|group_size|bits` (`**/benchmark/soar/demo_sala/prepare_env.sh`), 34 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#180-180), lines 180 to 320

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#320-320), lines 320 to 417

我同意你的判断。先不纠结下载差异，按你给的信息处理：公开评测集、公开评测脚本、本地这份 perf_public_set.jsonl 都视为与官方一致。

基于当前实现，除了把 SOAR_GPTQ_CALIBRATION_SAMPLES 从 32 提到 64，最有价值的不是再加样本数，而是改 64 条样本的组成方式。

先说当前代码里真正发生了什么：

1. 当前校准样本是顺序截断，不做打乱、不做分层。
在 preprocess_model.py 里，load_calibration_texts 只是从上往下读，拿到前 N 条就停。

2. 当前校准只读取 question，不读取 gold。
在 preprocess_model.py 里，默认 calibration_field 是 question。这个设置我建议保持，不要把答案喂进校准。

3. 当前 perf_public_set.jsonl 是按任务块排布的，不是随机混排。
所以前 64 条不等于“代表性的 64 条”，更接近“前面一段任务分布的 64 条”。

这就解释了为什么 64 不一定比 32 好：如果样本结构仍然偏，GPTQ 看到的激活分布还是不对。

我按收益和风险给出建议，优先级从高到低：

**第一优先级：改样本选择，不改量化参数**
这一步最值得做，而且不需要先改源码。你完全可以先手工生成一个新的 calibration jsonl，然后让 SOAR_GPTQ_CALIBRATION_FILE 指过去。

推荐两种选法：

1. 任务分层抽样
建议不要直接取前 64 条，而是按 task 分层。
一个实用起点：
- 24 条 mcq
- 40 条 niah

如果你更担心长上下文精度，可以再激进一点：
- 16 条 mcq
- 48 条 niah

原因：
- 4bit 后最容易伤的是长上下文/检索类激活分布
- 你前面的症状也更像“量化后长上下文稳态退化”，不太像纯短题退化

2. 长度分桶抽样
perf_public_set.jsonl 里本身有 prompt_tokens 字段，可以直接利用。
比起“任务均匀”，我更建议“任务 + 长度”同时分层。

一个更像样的 64 条组合可以是：
- 16 条短 mcq
- 16 条 30k 左右 niah
- 16 条 60k 左右 niah
- 16 条 120k 左右 niah

理由：
- GPTQ 校准看的是激活范围
- MiniCPM-SALA 这种长上下文模型，30k、60k、120k 的注意力统计差异很大
- 只拿短题或只拿前一段长题，不能覆盖真实部署分布

**第二优先级：去重或降权近重复模板**
这份数据里一些 niah 样本是同模板换关键词、换 magic number、换 uuid。
如果 64 条里塞太多同模板，只会让某一类激活被重复强调。

建议：
- 同一类 magic-number / magic-uuid 模板不要占太多
- 同一篇长上下文底文派生出的多个问题，不要重复选太多条

简单说，校准集要追求覆盖面，不要追求“都长得差不多”。

**第三优先级：试 group_size，不要只盯 calibration_samples**
当前是 group_size=128，在 preprocess_model.py 和 preprocess_model.py 这条路径里生效。

如果目标是提正确率，通常更值得试的是：
- 4bit, group_size=64
- 再不行才考虑更激进的保精度方案

经验上：
- 从 128 改到 64，经常比 32 样本改 64 样本更能实质改善准确率
- 代价是模型体积和部分 kernel 效率会差一点
- 这是典型的精度换吞吐，需要你再跑一次速度

如果你现在的痛点是正确率而不是极限速度，这个杠杆比继续堆样本更有希望。

**第四优先级：保持 calibration_field=question，不建议喂 gold**
这个我建议很明确：
- 不要把 gold 拼进校准文本
- 不要做 question + answer 的伪监督式校准

原因不是规则问题，而是数值分布问题：
- 你服务时真实输入是 question，不是 gold
- 校准应该拟合推理时输入激活，而不是人为改变激活分布
- 这样做短期可能让公开集看起来更好，但泛化更差

**第五优先级：SOAR_GPTQ_ATTN_IMPL 可以试，但我不把它当主线**
当前 preprocess 会把 SOAR_GPTQ_ATTN_IMPL 传给 GPTQModel.load，在 preprocess_model.py 附近。

这个旋钮可以试，但我对它的预期不高。
它更像数值路径一致性的小修正，不像决定性精度杠杆。

如果要试，建议只做两个点：
- flash_attention_2
- 一个更保守、和推理更接近的替代实现

但这条我会放在：
- 分层抽样之后
- group_size=64 之后

**第六优先级：当前 layer_aware 不太像真正能救精度的主因**
你现在的 layer_aware 本质上是通过 dynamic_rules 排除模块，在 preprocess_model.py 可以看到，当前 _build_dynamic_rules 实际上只用 exclude_modules，基本不参与更精细的“按层调策略”。

所以：
- SOAR_GPTQ_LAYER_AWARE 先保持 1
- 但不要对它抱太高期望
- 真正决定效果的还是“校准样本分布”和“group_size”

**我不建议优先做的事**
1. 继续只改 calibration_samples
你已经试过 64 没提升，这很符合当前代码逻辑，因为问题大概率不是样本数不够，而是样本分布不对。

2. 把 self_attn.o_gate / self_attn.z_proj 拉进量化
这条之前已经查过，风险高，收益不明确，不适合作为当前恢复正确率的主线。

3. 先去改 bits
从 4bit 改别的位宽会把实验面拉太大。除非你准备接受明显更大的速度/体积变化，否则先别动。

**如果按“最小风险、最高信息量”做实验，我建议顺序是**
1. 不改代码，手工做一个 64 条分层 calibration jsonl
2. 再做一个 64 条长度分桶 calibration jsonl
3. 在更优的那个样本集上试 group_size=64
4. 只有前三步都没改善，再考虑 attention impl 或模块覆盖率

**我会给你的具体判断**
就当前代码看，最可能提升正确率的不是：
- 64 比 32
而是：
- 哪 64 条
其次才是：
- group_size 128 改 64

如果你愿意继续往前走，下一步我建议先做一个正式优化提案，不直接改代码，内容是：

- 目标：把校准采样从顺序前 N 改成可控的分层/分桶采样
- 预期收益：提升 4bit 量化后正确率稳定性，优先改善长上下文退化
- 规则合规：仅使用公开数据集做离线校准，不引入隐藏集信息
- 风险：可能轻微牺牲部分短题拟合，或因样本分层不当导致收益不稳定
- 变更点：
  - preprocess_model.py
  - 新增可选环境变量控制采样模式、随机种子、任务配额、长度分桶
- 验证命令：
  - 量化命令
  - 本地 eval_model.py 正确率命令
  - 本地 run_soar_suite.py 速度命令
- 成功标准：
  - ori_accuracy 回升
  - 速度损失可控
- 回滚：
  - 恢复顺序前 N 采样

如果你要，我下一条就按这个仓库要求，给你一版完整的提案，不直接改代码。
