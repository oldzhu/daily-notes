对，这个方向是对的，而且比继续盲猜 calibration subset 更有信息量。

为什么有用
- 现在我们已经基本确认主问题在量化后的精度分布，不在 kernel。
- 仅看总的 `ori_accuracy` / `overall_accuracy`，看不出到底是：
  - `mcq` 掉得多
  - `qa` 掉得多
  - 还是长上下文桶更容易崩
- 如果你在 eval_model.py 的副本上加：
  - 分任务准确率
  - 分长度桶准确率
  - 最好再加 `task x length_bucket`
  那就能直接看到“错误主要集中在哪些区域”。

这对校准集选择的帮助
- 如果 `qa + 32k_128k` 掉得最厉害，说明 calibration 更该补这类语义连续、长上下文样本。
- 如果 `mcq + 0_4k` 基本稳定，而 `niah + 32k_128k` 很差，就不该再平均加样本，而要决定：
  - 是不是需要专门补这类样本
  - 还是这类任务本身不值得为了它牺牲整体精度
- 也就是说，这个统计不能直接告诉你“最优 calibration set 是什么”，但能把搜索空间从“全靠猜”缩到“针对具体退化区域做定向补样”。

但有两个重要 caveat
1. 这只是“诊断工具”，不是最终裁判
- 我之前看过你本地的 eval_model.py，它对部分长上下文多答案任务的打分偏宽松。
- 所以新增的分桶统计很适合看“相对变化趋势”，不适合把每个桶的绝对值当成官方真值。

2. 不要对 public set 过拟合
- 如果你完全按 public set 上最差桶去手工补样，容易把 calibration 调成“只会做 public set 分布”。
- 更稳的做法是：
  - 先用分桶统计找主要退化类型
  - 再补“同类型但更有代表性”的样本
  - 不要只补公开集里最难的那几条

我建议你加的统计最少包括这些
- `task`
- `length_bucket`
- `task_length_bucket`
- 每桶：
  - `count`
  - `correct`
  - `accuracy`
- 最后保留原有兼容输出：
  - `ori_accuracy`
  - `overall_accuracy`

最好再额外加一个机器可读输出
- 例如最后打印一行 JSON：
```text
bucket_accuracy={"task": {...}, "length_bucket": {...}, "task_length_bucket": {...}}
```
这样你后面做多次实验时可以直接抓日志对比，不用手工抄表。

结论
- 你的判断是对的。
- 这类分布统计很适合当前阶段，用来决定“哪些任务类型/长度段值得优先进入 calibration set”。
- 它的价值主要是定位退化区域，而不是直接给出最终答案。

如果你要我做这件事，按仓库流程我下一步会先给你一个正式改动提案，然后再去：
- 复制成 benchmark/soar/demo_sala/eval_model_001.py
- 保持 `ori_accuracy` / `overall_accuracy` 兼容
- 新增分任务、分长度桶、以及 `task x length_bucket` 统计输出
