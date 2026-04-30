basde on the latest 3 officail results as the below:
...
V19
Score
{
  "acc": 97.58,
  "acc_ori": 78.07,
  "final_score": 35.37,
  "benchmark_duration": {
    "S1": 591.04,
    "S8": 1113.65,
    "Smax": 2917.74
  }
}
v18:
Score
{
  "acc": 100.0,
  "acc_ori": 80.13,
  "final_score": 39.01,
  "benchmark_duration": {
    "S1": 586.64,
    "S8": 1087.9,
    "Smax": 2857.64
  }
}
the previous prev-18
Score
{
  "acc": 99.06,
  "acc_ori": 79.24,
  "final_score": 39.25,
  "benchmark_duration": {
    "S1": 596.25,
    "S8": 1066.35,
    "Smax": 2746.62
  }
}
...
basically I agree to use the v18 as the new base line with the below concerns.:
1. different s8/smax benchmark betwen the fcloud and official site. In fcloud v18 s8/smax better than pre-v18, but in official site v18 s8/smax worse than pre-v18.
2. use v18 as baseline, so we will revert to v18 commit in repo and then do the next optimization from this v18 commit point?
3. I think the next thing is kv cache mixed nvfp4 and fp8 as one champion  mentioned as the below:
...
02 NVFP4 KV Cache
NVFP4 采用 FP4 E2M1 格式（1 位符号 + 2 位指数 + 1 位尾数），两个 FP4 值打包存储在一个 uint8 字节中，相比 FP8 KV Cache 再减少一半的存储与带宽开销。每个量化 block 配一个 scale factor，反量化时乘以 scale 即可恢复数值。
这种格式非常适合 decode 阶段的 memory-bound 特性：KV Cache 的读取量直接减半，对于长序列场景收益尤为明显。我们在 SGLang 框架中适配了完整的 NVFP4 KV Cache 链路，包括 FP4 内存池管理、注意力后端适配以及量化写入与反量化读取路径的打通。
启用后速度确实有提升，但随之而来的问题是精度下降明显，相比 FP8 KV Cache 约 80% 的正确率，纯 NVFP4 KV Cache 的正确率掉到了约 75%，已经低于可接受的阈值。
03 混合精度 KV Cache
纯 FP4 精度不够，纯 FP8 速度不够，我们很自然地想到用混合精度，评估每一层 K/V 值对最终输出精度的敏感程度，对精度敏感的层保留 FP8 KV Cache，对精度不敏感的层使用 NVFP4 KV Cache。具体做法是逐层做消融实验，依次将某一层的 KV Cache 从 FP8 降为 FP4，观察模型输出精度的变化幅度，以此判断该层的重要性。
实验发现，模型首层和末尾几层对量化精度最为敏感，中间层则表现出较强的鲁棒性。最终我们将少数关键层保留为 FP8，其余层全部使用 NVFP4，在几乎损失精度较小的前提下，获得了接近纯 FP4 的带宽收益。
...


