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
