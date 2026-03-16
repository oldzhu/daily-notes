I submitted and now I got my 2nd official score as the below:
the 2nd official score:
 "acc_ori": 80.07,"S1": 458.78,"S8": 634.35,"Smax": 1140.66
 my prevous 1st score is:
 the 1st official scope 
"acc_ori": 80.67 "S1": 539.46,"S8": 747.59,"Smax": 1365.8
Now I am from 10th to 6th, good improving and we need do more to move into top 3 or top 5.
pls do more research on the current code path(fp8+gpugraph+denseminicpm) to see if there are part of code(prefill or decode) we can do more improving on speed meanwhile keep the goot acc_ori. is it time to move to fp8+gpugraph+sparse or statr at 路径二：投机采样. I will put the two improving paths mentioned by officail owner as the below:
...
路径一：量化加速
可选路径：GPTQ W4A16 + Marlin Kernel + FP8 KV Cache

将模型权重量化为 4-bit（W4A16），利用 Marlin 高性能反量化 GEMM Kernel 加速 Linear 计算；KV Cache 量化为 FP8 减少 Decode 阶段显存带宽瓶颈。量化工具（GPTQModel）开箱即用，SGLang 对 GPTQ + Marlin 支持完善，选手只需提交量化脚本在评测机上现场量化。

工作流程：

编写量化脚本： 使用 GPTQModel 对 SALA FP16 权重做 W4A16 量化（group_size=128），脚本作为提交物。
验证正确性： --quantization gptq_marlin 启动，确认 accuracy > 97%；掉点严重可回退 W8。
开启 KV Cache FP8： --kv-cache-dtype fp8_e5m2，长上下文场景收益显著。注意 Lightning Attention 层使用独立线性注意力状态，优化路径不同。
进阶调优： 针对 6000D 调整 Marlin Kernel 的 tile / warp 配置。
可能需要阅读和修改的核心文件：

模型与加载
python/sglang/srt/models/minicpm_sala.py — SALA 模型定义
python/sglang/srt/model_loader/loader.py、weight_utils.py — 权重加载与量化映射
python/sglang/srt/server_args.py — 启动参数（--quantization、--kv-cache-dtype）
量化方法
python/sglang/srt/layers/quantization/gptq.py — GPTQ 线性层与 Marlin 调度
python/sglang/srt/layers/quantization/__init__.py — 量化方法注册表
python/sglang/srt/layers/linear.py — 主 Linear 层，量化方法注入点
CUDA 算子
sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu、marlin_template.h — Marlin W4A16 GEMM Kernel
sgl-kernel/csrc/gemm/gptq/gptq_kernel.cu — GPTQ 反量化 Kernel
KV Cache 量化
python/sglang/srt/layers/quantization/kv_cache.py — KV Cache 量化逻辑
路径二：投机采样
可选路径：EAGLE3 多层 Draft Head（需算法创新）

EAGLE3 通过轻量 Draft Head 利用目标模型隐藏状态预测候选 token，再由目标模型一次性验证。Head 参数量极小（几十 MB），满足 2GB 限制。

Lightning Attention 兼容性挑战： SALA 部分层使用 Lightning Attention（Gated Delta Rule 线性注意力），其递推计算本质上不支持树状因果掩码，传统树验证机制无法直接生效。SGLang 已有初步集成（hybrid_linear_attn_backend.py 中处理了 is_target_verify 模式），但选手仍可能需要在算法层面创新，鼓励创新方案。

工作流程：

训练 EAGLE3 Head： 参考 EAGLE 仓库，用 SALA 收集隐藏状态训练 Draft Head，权重作为提交物。
模型适配： 参考 llama_eagle3.py 为 SALA 创建 EAGLE3 模型文件。
解决 Lightning Attention 验证问题： 核心难点，需修改验证逻辑适配线性注意力层。
启动验证： --speculative-algorithm EAGLE3 --speculative-draft-model-path <path> --speculative-num-draft-tokens 5。
可能需要阅读和修改的核心文件：

EAGLE3 Pipeline
python/sglang/srt/speculative/multilayer_eagleworker.py — EAGLE3 Draft Worker 主循环
python/sglang/srt/speculative/eagle_utils.py — 树掩码构建与 Verify 函数（适配线性注意力的重点）
python/sglang/srt/speculative/eagle_info.py — Verify / Draft 数据结构
python/sglang/srt/speculative/multi_layer_eagle_utils.py — EAGLE3 Triton Kernel
模型适配
python/sglang/srt/models/llama_eagle3.py — 参考模板：LLaMA EAGLE3 实现
python/sglang/srt/models/minicpm_sala.py — SALA 目标模型（需据此创建 EAGLE3 版本）
Lightning Attention（理解验证兼容性的关键）
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py — 混合线性注意力 Backend（已有初步 spec 集成）
python/sglang/srt/layers/radix_linear_attention.py — 线性注意力层接口
python/sglang/srt/layers/attention/fla/chunk.py、fused_recurrent.py — 线性注意力 Triton Kernel
python/sglang/jit_kernel/cutedsl_gdn.py — GDN CUDA Kernel
CUDA 算子
sgl-kernel/csrc/speculative/eagle_utils.cu — 树构建 + Verify Kernel
sgl-kernel/csrc/speculative/speculative_sampling.cu — Sampling Kernel
...
can we still doing something on "1.注意 Lightning Attention 层使用独立线性注意力状态，优化路径不同。
2.进阶调优： 针对 6000D 调整 Marlin Kernel 的 tile / warp 配置"?
I put one hardware optimize comments for rtx pro sm120(nvidia 6000d which is out official gpu used to test and evaluate) into ~/sglang/benchmark/soar/data and you can refer when plan/consider hardware related optimizations.  
