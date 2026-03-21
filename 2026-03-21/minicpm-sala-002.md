我去核对当前提交链路里的量化格式、group 定义和运行时 kernel，判断它和冠军文里说的 NVFP4 是不是同一类方案。

Searched for regex `NVFP4|nvfp4|E2M1|E4M3|gptq_marlin|GPTQModel|QuantizeConfig|qweight|scales|qzeros|group_size` (`**/benchmark/soar/demo_sala/**`), 200 results

Searched for regex `NVFP4|nvfp4|fp4|gptq_marlin|marlin|gptq|group_size|qzeros|qweight|scales` (`**/python/sglang/**`), 200 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#180-180), lines 180 to 280

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh#34-34), lines 34 to 60

不是，我们当前这套量化代码和冠军文里说的 NVFP4 不是同一种方案。

**结论先说**
你现在的提交链路是：

- 离线量化：GPTQModel
- 推理加载：`--quantization gptq_marlin`
- 权重量化形态：GPTQ W4A16
- 默认 `group_size=128`

而冠军文里说的是：

- NVFP4 权重量化
- 4-bit 浮点格式，E2M1
- 每 16 个权重一个局部组
- 局部 FP8 E4M3 scale
- 整个 tensor 再有全局 FP32 scale

这两者不是一个格式，也不是同一条工具链。

**我们当前代码实际在做什么**

从 preprocess_model.py 开始，当前量化路径明确是：

- `from gptqmodel import GPTQModel, QuantizeConfig`

然后在 preprocess_model.py 用的是：

- `QuantizeConfig(bits=bits, group_size=group_size, dynamic=dynamic_rules)`

这就是标准 GPTQ 配置入口，不是 NVFP4。

默认参数在 prepare_env.sh 到 prepare_env.sh：

- `SOAR_GPTQ_BITS=4`
- `SOAR_GPTQ_GROUP_SIZE=128`

服务端加载在 prepare_env.sh：

- `--quantization gptq_marlin`

所以我们当前实际是：

- GPTQ 4bit
- Marlin kernel
- group size 128
- 不是 NVFP4

**为什么我能确定它不是 NVFP4**

因为当前代码里明显是 GPTQ/Marlin 的数据结构和路径。

你可以看到仓库里到处都是 GPTQ/Marlin 相关痕迹，比如：
- `qweight`
- `scales`
- `qzeros`
- `gptq_marlin`

这类结构在检索结果里都出现了。  
而冠军描述的 NVFP4 是另一种 4-bit 浮点权重格式，不是 GPTQ 的 `qweight + qzeros + scales` 这套表示。

更直接一点：

- GPTQ W4A16 本质上是“压缩后的 4bit 权重 + scale/zero-point/dequant kernel”
- NVFP4 本质上是“4-bit float 权重格式本身”，且冠军提到的是 Blackwell 原生加速路径

这不是同一件事。

**仓库里有没有 NVFP4 能力**
有，但我们没在用。

从检索结果看，SGLang 本身支持一些 NVFP4 相关路径，例如：
- `nvfp4`
- `petit_nvfp4`
- 相关环境变量
- server args 里也有 `nvfp4` 选项

这说明“框架支持 NVFP4”这件事是存在的。

但你的 demo_sala 提交链路没有走那条路。当前这条链路没有：
- 生成 NVFP4 checkpoint
- 在 prepare_model.py 里做 NVFP4 格式转换
- 在 prepare_env.sh 里以 `--quantization nvfp4` 启动

所以答案是：

- 框架可能支持
- 我们当前提交方案没用

**和冠军方案的关键差异**

1. 量化格式不同
- 我们：GPTQ 4bit
- 他：NVFP4

2. 分组粒度不同
- 我们默认：`group_size=128`
- 他描述：每 16 个权重一组局部缩放

3. 缩放机制不同
- 我们：GPTQ scale/zero-point 风格
- 他：局部 FP8 E4M3 scale + 全局 FP32 scale

4. 推理 kernel 路径不同
- 我们：`gptq_marlin`
- 他更像是 Blackwell 原生 NVFP4 加速链路

**这对我们当前 accuracy 问题意味着什么**

这件事很重要，因为它说明：

- 你现在看到的 accuracy unstable，不一定能直接从冠军的 NVFP4 经验里照搬解决
- 因为你们不是同一种量化格式

但冠军那篇分享里有两部分经验仍然对我们高度相关：

1. 校准数据集组成很关键
2. 低语义、随机 UUID / keyword 类样本不宜在校准集中占比过高

这两点和量化格式无关，对 GPTQ 也成立。

反过来，和 NVFP4 强绑定的部分，比如：
- E2M1
- 每 16 权重一组
- 局部 FP8 + 全局 FP32 双层缩放

这些目前并不是我们代码里正在做的事。

**我的判断**
如果你问“我们当前量化代码是不是已经和冠军用的是同一种量化方案”，答案是：

- 不是

如果你问“冠军的分享对我们有没有参考价值”，答案是：

- 有，但主要参考价值在校准数据分析和精度稳定性治理
- 不在于直接照搬 NVFP4 量化格式

**接下来怎么用这个信息**
在你做 `sequential + 8/16` 回放时，我建议保持清醒一点：

1. 不要把“冠军用了 NVFP4”理解成“我们当前 GPTQ 配置一定错了”
2. 更合理的理解是：
   - 他的格式更先进
   - 但我们当前能立刻借鉴的是他的评测和校准方法论
3. 如果后续真的想追 NVFP4，那会是另一条独立 feature，不是当前这条 GPTQ 校准修复线里的小修小补

1. 你先继续跑 `sequential + 8/16`
2. 跑完后我可以基于“我们不是 NVFP4，而是 GPTQ W4A16”这个前提，帮你重新判断哪些冠军经验能直接迁移，哪些不能
