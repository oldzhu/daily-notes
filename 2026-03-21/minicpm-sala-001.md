这是对的，应该先把基线钉住，再继续试量化。

你现在最值得保留的三组基线是：

1. 非量化基线
- 用当前服务路径但不走 GPTQ
- 这给你一个“模型本体 + 当前推理栈”的正确率上限参考

2. `sequential + 8`
- 用来验证你记忆里的 last-good 小样本校准路径

3. `sequential + 16`
- 用来判断“稍微多一点顺序样本”是否还能保持那个较好的精度区间

我建议你把每次实验都固定记录成同一个表，不然很容易混淆：

- quant mode
- calibration sampling
- calibration samples
- group size
- attn impl
- ori_accuracy
- overall_accuracy
- total duration
- 备注：是否是重复 run、是否有异常输出长度

最重要的是，先不要同时改别的变量。  
也就是在你做这轮回放时，尽量固定：

- prepare_env.sh
  里的 `SOAR_GPTQ_GROUP_SIZE`
- prepare_env.sh
  里的 `SOAR_GPTQ_ATTN_IMPL`
- preprocess_model.py
  当前模块覆盖逻辑
- 服务启动参数也不要改

这样你得到的结论才有用：

1. 如果 `sequential + 8/16` 明显优于现在的 stratified
- 说明主要问题确实在校准样本组成
- 下一步就该做“语义过滤的采样”，而不是继续堆样本数

2. 如果 `sequential + 8/16` 也回不去
- 说明问题不只是 sampling
- 那就该优先查 group_size、量化链路版本、或者别的数值路径差异

你提到“也测试了非量化模型分数，之后可以 revert back”，这个思路是对的。  
更准确地说，非量化结果不是为了回滚代码，而是为了定义两个参考线：

1. 上参考线：
- 当前非量化 correctness
- 告诉我们理论上现在这套服务栈能到哪里

2. 下参考线：
- 当前量化最差结果
- 告诉我们量化损失到底有多大

这样你后面看每一轮量化实验，不是只看绝对分数，而是看：

$$
\Delta_{\text{quant}} = \text{Accuracy}_{\text{bf16/non-quant}} - \text{Accuracy}_{\text{quant}}
$$

这个差值通常比单看 `ori_accuracy` 更有信息量。

我建议你这轮测试顺序就按这个来：

1. 非量化 baseline
2. `sequential + 8`
3. `sequential + 16`

如果时间够，再补：

4. `sequential + 32`

这样你能直接看出：
- 小样本顺序校准是否真的更稳
- 精度是不是随着样本数增加反而变差
- 最佳点是不是确实出现在 8 或 16

你跑完把这几组结果发我，我可以直接帮你判断下一步该走哪条：

1. 回退到小样本顺序校准
2. 做“语义过滤校准”新 feature
3. 还是转去试 `group_size=64`
