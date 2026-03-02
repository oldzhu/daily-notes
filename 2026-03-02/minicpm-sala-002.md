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
