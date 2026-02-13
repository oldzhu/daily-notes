Excellent. This is a comprehensive and very practical topic. Based on the search results, I have synthesized the landscape of LLM optimization across CPU and GPU platforms. Below is a categorized framework covering **Training** and **Inference** for both hardware types, mapping each technique to the specific part of the algorithm or computation it targets.

这是一个非常全面且实用的主题。根据搜索结果，我整合了CPU和GPU平台上LLM优化的全景图。以下是一个分类框架，涵盖两种硬件类型的**训练**和**推理**，将每种技术映射到其针对的特定算法或计算环节。

---

# 📋 Optimization Categories / 优化分类目录
**English & Chinese / 英文与中文**

| Hardware / 硬件 | Stage / 阶段 | Optimization Technique / 优化技术 | Target Part of Algorithm / 算法目标环节 | Simple Summary / 简述 |
| :--- | :--- | :--- | :--- | :--- |
| **GPU** | **Training** | **Activation CPU Offloading**  | Backward Pass (Memory Footprint) / 反向传播（内存占用） | Moves intermediate activations from GPU RAM to CPU RAM to enable larger batch sizes; trades speed for memory. / 将中间激活张量从GPU显存移至CPU内存，以支持更大批量；以速度为代价换取内存空间。 |
| **GPU** | **Training** | **Unified Memory (UM)**  | Data Transfer Management / 数据传输管理 | Simplifies programming by automatically migrating data between CPU and GPU; performance depends on access patterns (good for LoRA, bad for full tuning). / 通过CPU与GPU间自动数据迁移简化编程；性能取决于访问模式（对LoRA友好，对全量微调不友好）。 |
| **GPU** | **Training** | **Automatic Mixed Precision (AMP)**  | Matrix Multiplications (GEMM) / 矩阵乘法 | Uses FP16/BF16 for compute, FP32 for master weights; leverages Tensor Cores for 2-4x speedup. / 使用FP16/BF16进行计算，FP32保存主权重；利用Tensor Core实现2-4倍加速。 |
| **GPU** | **Training** | **FP8 Training**  | Matrix Multiplications (GEMM) / 矩阵乘法 | Extreme low-precision (8-bit) training on Hopper/Blackwell; requires Transformer Engine, significantly reduces memory and increases TFLOPs. / 在Hopper/Blackwell架构上的极端低精度（8位）训练；需Transformer Engine，显著降低显存并提升算力。 |
| **GPU** | **Training** | **QLoRA / 4-bit Quantization**  | Weight Storage (Fine-tuning) / 权重存储（微调） | Loads model in 4-bit (NF4) and adds adapters; enables 40B+ fine-tuning on single consumer GPU (70% VRAM reduction). / 以4位精度加载模型并添加适配器；实现单张消费级GPU微调400亿参数模型（显存减少70%）。 |
| **GPU** | **Training** | **Custom Triton Kernels**  | Attention & Projection Layers / 注意力层与投影层 | Hand-written kernels (Unsloth) to reduce backward pass computation and memory writes. / 手写内核减少反向传播计算量与内存写入（如Unsloth）。 |
| **GPU** | **Inference** | **FlashInfer/Fused Kernels**  | Attention (KV Cache) & MoE / 注意力机制与混合专家 | Fuses multi-step operations (e.g., RoPE+Q+Cache) to reduce memory round-trips and launch overhead. / 融合多步操作以减少内存往返与内核启动开销。 |
| **GPU** | **Inference** | **FP8 KV Cache**  | Key-Value Cache Storage / 键值缓存存储 | Stores KV cache in 8-bit precision; increases concurrent request capacity without heavy accuracy loss. / 以8位精度存储KV缓存；在不严重损失精度前提下提升并发请求容量。 |
| **GPU** | **Inference** | **torch.compile Graph Fusion**  | Compute Graph / 计算图 | Automatically fuses operations (e.g., AllReduce + RMSNorm); reduces kernel launch frequency. / 自动融合操作（如全规约+RMSNorm）；减少内核启动频率。 |
| **GPU** | **Inference** | **Async Scheduling**  | Host-side Batching / 主机端批处理 | Decouples CPU request scheduling from GPU execution; hides host overhead (critical for fast GPUs like Blackwell). / 将CPU请求调度与GPU执行解耦；隐藏主机开销（对Blackwell这类高速GPU至关重要）。 |
| **GPU** | **Inference** | **Stream Interval**  | Network I/O / 网络输入输出 | Buffers tokens before sending; reduces CPU serialization/HTTP overhead (up to 57% gain). / 缓冲token后再发送；降低CPU序列化/HTTP开销（最高提升57%效率）。 |
| **GPU** | **Inference** | **MoE Kernel Selection**  | Mixture-of-Experts Routing / 混合专家路由 | Specifically offloads "expert" layers to CPU while keeping attention layers on GPU; maximizes VRAM utility. / 特指将“专家”层卸载至CPU，同时保留注意力层在GPU；最大化显存利用率。 |
| **GPU** | **Inference** | **Tensor Parallelism (TP)**  | Layer Partitioning / 层切分 | Splits weights across GPUs; communication via AllReduce. / 将权重切分至多GPU；通过全规约通信。 |
| **GPU** | **Inference** | **Disaggregation**  | Prefill/Decode Separation / 预填充与解码分离 | Runs prefill and decode on separate GPUs; optimizes compute/memory ratio per stage. / 将预填充与解码阶段分离至不同GPU；优化各阶段计算/内存配比。 |
| **CPU** | **Training** | **LoRA (Parameter-Efficient)**  | Weight Updates / 权重更新 | Freezes base weights; trains only small adapters; avoids swap thrashing on RAM-constrained systems. / 冻结基座权重；仅训练小型适配器；避免内存受限系统下的频繁交换。 |
| **CPU** | **Training** | **Full Model CPU Offloading**  | Entire Model State / 完整模型状态 | When GPU VRAM is insufficient, entire model resides on CPU; execution happens via CPU or page-migration. / 当GPU显存不足时，完整模型驻留CPU；通过CPU执行或页迁移运行。 |
| **CPU** | **Inference** | **SmoothQuant (8-bit)**  | Activation Outliers / 激活异常值 | Smooths outliers from activations to weights; enables INT8 without accuracy loss on Xeon. / 将激活中的异常值迁移至权重；使INT8量化在至强CPU上无精度损失。 |
| **CPU** | **Inference** | **Ultra-low-bit (1/1.58/2-bit)**  | GEMM Microkernels / 通用矩阵乘微内核 | Custom AVX2/AMX microkernels for binary/ternary weights; up to 7x speedup vs FP16. / 针对二值/三值权重的自定义AVX2/AMX微内核；相比FP16最高7倍加速。 |
| **CPU** | **Inference** | **Selective MoE Offloading**  | Expert Layers (MoE) / 专家层 | Only offloads MoE expert weights to RAM; critical attention layers stay on GPU (or iGPU). / 仅将MoE专家权重卸载至内存；关键注意力层保留在GPU。 |

---

# 🧠 Next Steps: Deep Dive Roadmap / 下一步：深度探讨路线图

We now have the map. Based on the strength of the search results, I suggest we proceed in the following logical order. This sequence moves from **memory-centric** solutions to **compute-centric** and finally **system-architecture** solutions.

我们现在有了完整的地图。根据搜索结果的详实程度，我建议按照以下逻辑顺序逐一深入。这个顺序是从**内存为中心**的解决方案，到**计算为中心**，最后是**系统架构**级方案。

## 🔹 Track 1: Memory Wall & Capacity (The VRAM Crisis) / 内存墙与容量（显存危机）
*Focus: How to fit 100B models into 24GB GPUs or DRAM.*
*重点：如何将千亿参数模型塞进24GB显存或内存中。*

1.  **Quantization Spectrum** (GPU/CPU) / 量化全谱
    - *From SmoothQuant 8-bit on Xeons to MXFP4/NVFP4 on Blackwell, down to 1-bit BitNet on AI PCs.*
    - *从至强的SmoothQuant 8-bit，到Blackwell的MXFP4/NVFP4，再到AI PC的1位BitNet。*
    - **Key Sources**: 
2.  **CPU Offloading Strategies** / CPU卸载策略
    - *Generic layer-splitting vs. Selective MoE offloading. Why MoE is special?*
    - *通用层切分 vs 选择性MoE卸载。为什么MoE特殊？*
    - **Key Sources**: 
3.  **Parameter-Efficient Fine-Tuning (PEFT)** / 参数高效微调
    - *Why LoRA succeeds on CPU-only training while Full Tuning fails (swap thrashing).*
    - *为什么LoRA在纯CPU训练中成功，而全量微调失败（交换死锁）。*
    - **Key Sources**: 

## 🔹 Track 2: The Math Engine (Kernels & Precision) / 数学引擎（内核与精度）
*Focus: Making GEMM and Attention faster.*
*重点：让矩阵乘法与注意力计算更快。*

1.  **Low Precision Training** / 低精度训练
    - *AMP (FP16) -> FP8 -> FP4. The role of Tensor Cores and Transformer Engine.*
    - *从混合精度到FP8到FP4。Tensor Core与Transformer Engine的角色。*
    - **Key Sources**: 
2.  **Attention & MoE Kernel Wars** / 注意力与MoE内核之争
    - *FlashInfer, Cutlass, trtllm-gen. Why "Fusion" reduces memory bound.*
    - *FlashInfer, Cutlass, trtllm-gen。为什么“融合”能减少内存瓶颈。*
    - **Key Sources**: 

## 🔹 Track 3: System & Overhead (Hiding the CPU) / 系统与开销（隐藏CPU）
*Focus: When the GPU is too fast, the CPU becomes the bottleneck.*
*重点：当GPU太快时，CPU成为瓶颈。*

1.  **Host Overhead Hiding** / 主机开销隐藏
    - *Async Scheduling, Stream Interval, Continuous Batching evolution.*
    - *异步调度、流间隔、连续批处理演进。*
    - **Key Sources**: 
2.  **Graph Capture & Compilation** / 图捕获与编译
    - *torch.compile fusions vs. manual kernel writing.*
    - *torch.compile融合 vs 手动写内核。*
    - **Key Sources**: 

## 🔹 Track 4: Architecture Shift / 架构变革
*Focus: Breaking the single-GPU, single-stage paradigm.*
*重点：打破单GPU、单阶段范式。*

1.  **Disaggregated Inference** / 分离式推理
    - *Separating Prefill (compute-bound) from Decode (memory-bound).*
    - *分离预填充（计算密集型）与解码（内存密集型）。*
    - **Key Sources**: 
2.  **Data+Expert Parallelism (DEP)** / 数据+专家并行
    - *The next frontier for MoE scaling.*
    - *MoE扩展的下一个前沿。*
    - **Key Sources**: 

---

**Please choose which Track or specific technique you would like to start with.** The search results provide particularly rich, actionable data on **CPU Ultra-low-bit inference** (Track 1) and **vLLM Blackwell Optimizations** (Track 3/4), but I am ready to dive into any of them with architectural diagrams and kernel-level explanations.

**请选择您想从哪个专题或具体技术开始。** 搜索结果中关于**CPU超低位推理**（专题一）和**vLLM Blackwell优化**（专题三/四）提供了非常丰富且可操作的实践数据，我可以随时结合架构图和内核级解释进行深入讲解。
