由于FCloud实例暂时无法访问，我们可以先详细规划后续各个阶段的优化方案。这样一旦环境恢复，你就可以按照计划有条不紊地推进。下面我将逐一展开每个阶段的详细计划、技术要点和关键代码示例。

---

## **第一阶段回顾：基准建立（已完成）**

你已经完成了：
- 环境搭建和模型部署
- 修复了短prompt的bug
- 验证服务可用

基准数据（正确性得分、各并发度下的`benchmark_duration`）待实例恢复后需立即采集。

---

## **第二阶段：GPTQ 4bit量化（详细实施方案）**

### **目标**
实现4bit权重量化，利用Marlin Kernel加速，保证正确性≥97%基准分，显著降低显存占用和推理延迟。

### **技术依据**
- SGLang原生支持GPTQ量化，通过`--quantization gptq_marlin`启动
- Marlin Kernel是专门针对4bit量化设计的高效GEMM内核，在NVIDIA GPU上性能优异
- GPTQModel提供了与SGLang无缝集成的量化工具

### **详细步骤**

#### **2.1 安装GPTQModel与依赖**
```bash
# 在虚拟环境中安装
source /root/sglang/sglang_minicpm_sala_env/bin/activate
pip install gptqmodel[sglang] --no-build-isolation
```
注意`--no-build-isolation`避免与现有环境冲突。

#### **2.2 准备校准数据**
从比赛公开数据集提取文本，或自行构造覆盖长上下文的样本。建议使用`perf_public_set.jsonl`中的问题字段。

```python
# extract_calib.py
import json
import random

with open('/path/to/perf_public_set.jsonl', 'r') as f:
    lines = [json.loads(line)['question'] for line in f]

# 选择512条，并确保长度多样性
calib_texts = random.sample(lines, min(512, len(lines)))
with open('calib.txt', 'w') as f:
    for text in calib_texts:
        f.write(text + '\n')
```

#### **2.3 编写量化脚本（prepare_model.sh的雏形）**
根据新提交要求，量化应在评测环境的`prepare_model.sh`中执行。下面是一个可独立运行的量化脚本，之后可移植到提交包中。

```python
# quantize_for_competition.py
import argparse
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='原始模型路径')
    parser.add_argument('--output', required=True, help='量化后模型保存路径')
    parser.add_argument('--calib', default='calib.txt', help='校准文本文件')
    args = parser.parse_args()

    # 加载校准文本
    with open(args.calib, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,      # 可调优参数
        sym=False,
        format="GPTQ"
    )

    model = GPTQModel.load(
        args.input,
        quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)

    model.quantize(
        texts,
        batch_size=2,
        calib_seq_len=2048
    )

    model.save(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"量化完成，模型保存至 {args.output}")

if __name__ == "__main__":
    main()
```

#### **2.4 测试量化模型**
启动服务并测试正确性：

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/quantized_model \
  --host 0.0.0.0 --port 30001 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 16384 \
  --quantization gptq_marlin   # 关键参数，启用Marlin内核
```

运行`eval_model.py`获取正确性得分，并与原始模型对比。

#### **2.5 调优技巧**
- 如果正确性下降过多，尝试：
  - 增大`group_size`到256（牺牲一点压缩率）
  - 设置`sym=True`
  - 添加更多、更多样化的校准数据
  - 使用`desc_act=False`（可能略微降低精度但提升速度）
- 如果速度不达标，检查是否真的调用了Marlin kernel（日志中应有相关提示），或尝试调整`--cuda-graph-max-bs`等参数。

---

## **第三阶段：KV Cache FP8量化**

### **目标**
将KV Cache量化为FP8，减少显存带宽占用，特别利好长上下文场景（比赛重点）。

### **技术依据**
- SGLang支持KV Cache FP8量化，通过`--kv-cache-dtype fp8_e5m2`启用
- 对于MiniCPM-SALA，需要注意稀疏注意力层和线性注意力层的不同处理（参考技术路径指引[citation:toolkit]）

### **详细步骤**

#### **3.1 确认模型兼容性**
检查`minicpm_sala.py`中对KV Cache的处理，确保FP8量化路径已实现。必要时可参考`kv_cache.py`中的实现。

#### **3.2 启动服务测试**
在量化模型基础上，增加FP8 KV Cache参数：

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/quantized_model \
  --host 0.0.0.0 --port 30002 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 16384 \
  --quantization gptq_marlin \
  --kv-cache-dtype fp8_e5m2    # 新增
```

#### **3.3 正确性与性能测试**
同样使用`eval_model.py`和`bench_serving.sh`测试，重点关注长上下文请求的耗时变化。

#### **3.4 可能遇到的问题及解决**
- **精度下降**：FP8动态范围有限，长上下文中KV值可能溢出。可尝试`fp8_e4m3`（如果支持），或调整缩放因子。
- **内核不支持**：确保SGLang版本包含FP8 KV Cache的kernel实现（需检查`kv_cache.py`和相应CUDA代码）。

---

## **第四阶段：算子优化与融合**

这是最硬核、也是拉开差距的关键。比赛明确指出“重点围绕推理优化（算子融合、Kernel优化、内存与KV读写优化、Prefill/Decode路径优化、图编译/算子调优等）”[citation:competition]。我们聚焦稀疏注意力部分的优化。

### **目标**
针对MiniCPM-SALA的**稀疏注意力层**，实现自定义融合kernel，减少内存访问和kernel launch开销。

### **分析瓶颈**
根据论文和初步profile，稀疏注意力主要耗时在：
1. 对长序列的**top-k选择**（计算每个query与所有key的相似度，取top-k）
2. 根据top-k索引**gather**对应的value
3. 对gather后的value做**加权求和**

目前这三个步骤可能由多个独立kernel完成，导致多次显存读写。

### **优化思路**
设计一个**融合kernel**：输入query、key cache、value cache，直接输出加权结果，中间结果（相似度、索引）保留在寄存器/共享内存中，避免全局内存往返。

### **实施计划**

#### **4.1 定位代码**
相关文件：
- `python/sglang/srt/layers/attention/minicpm_backend.py`：稀疏注意力的调度逻辑
- `sgl-kernel/csrc/sparse_attention/`（如果存在）或需要新建

#### **4.2 设计kernel接口**
```cuda
// 伪代码
void fused_sparse_attention_kernel(
    const float* query,        // [num_heads, head_dim]
    const float* key_cache,    // [seq_len, head_dim]
    const float* value_cache,  // [seq_len, head_dim]
    int seq_len,
    int k,                      // top-k数量
    float* output               // [num_heads, head_dim]
);
```

#### **4.3 实现步骤**
1. **加载query**到寄存器或共享内存
2. **分块遍历key cache**，计算点积，并维护一个大小为k的最小堆（存储score和index）
3. 遍历完成后，堆中即为top-k结果
4. 根据堆中的index从value cache中gather对应的value向量
5. 对gathered values进行加权求和（softmax权重可选，需根据模型设计）
6. 写回output

#### **4.4 性能调优**
- **块大小**：根据GPU架构（如RTX 6000D Ada）调整block size和grid size
- **使用向量化加载**：float4/int4等提高带宽利用率
- **共享内存优化**：key块可预取到共享内存
- **处理变长序列**：考虑padding或动态并行

#### **4.5 集成到SGLang**
在`minicpm_backend.py`中，当检测到自定义kernel可用时，调用它替代原有多个步骤。

#### **4.6 测试与验证**
- 单元测试：与原始实现输出对比，确保数值一致（允许小误差）
- 性能测试：使用`bench_serving.sh`对比优化前后延迟

---

## **第五阶段：提交包制作与最终调优**

### **5.1 理解提交结构**
根据最新要求[citation:toolkit]，提交包必须是`.tar.gz`，包含：
- `prepare_env.sh`（必须）：环境构建脚本
- `prepare_model.sh`（可选）：模型预处理脚本
- 其他代码（如自定义SGLang源码）

### **5.2 构建提交包**

#### **目录结构示例**
```
my_submission/
├── prepare_env.sh
├── prepare_model.sh
├── preprocess_model.py    # prepare_model.sh调用的量化脚本
└── sglang/                # 自定义SGLang源码（整个python目录）
    └── python/
        ├── sglang/...
        └── setup.py
```

#### **prepare_env.sh 内容**
```bash
#!/bin/bash
# 使用uv安装自定义sglang（editable模式）
uv pip install --no-deps -e ./sglang/python

# 设置服务启动参数（可被平台覆盖）
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 16384"

# 如果有其他依赖
uv pip install gptqmodel[sglang]
```

#### **prepare_model.sh 内容**
```bash
#!/bin/bash
# 接受 --input 和 --output 参数
input_dir=""
output_dir=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input) input_dir="$2"; shift 2;;
        --output) output_dir="$2"; shift 2;;
        *) echo "未知参数: $1"; exit 1;;
    esac
done

# 调用量化脚本
python preprocess_model.py --input "$input_dir" --output "$output_dir"
```

#### **preprocess_model.py**（集成量化逻辑）
将之前的量化脚本适配为从`--input`读取原始模型，输出到`--output`，并包含校准数据（需内嵌或从某处获取）。

### **5.3 本地测试提交包**
模拟平台环境：
```bash
# 创建测试目录
mkdir test_env
cp -r my_submission/* test_env/
cd test_env
# 模拟平台执行prepare_env.sh（source方式）
source prepare_env.sh
# 模拟平台调用prepare_model.sh
bash prepare_model.sh --input /path/to/original/model --output ./quantized_model
# 手动启动服务测试
python3 -m sglang.launch_server --model-path ./quantized_model ...
```

### **5.4 最终调优**
- **参数网格搜索**：对`group_size`、`desc_act`、`sym`等量化参数进行组合测试，找到正确性和速度的最佳平衡点。
- **并发度专项优化**：针对S1、S8、Smax三档并发，调整`--cuda-graph-max-bs`、`--chunked-prefill-size`等参数。
- **Profiling复查**：用Nsight Systems再次profile，确认瓶颈已解决。

---

## **讨论点**

1. **校准数据的选择**：我们能否使用`perf_public_set.jsonl`？它是否包含足够长的文本？可能需要额外生成长上下文样本。
2. **量化参数优先级**：先固定一组参数（如group_size=128, desc_act=True）进行第一轮测试，再根据结果调整。
3. **稀疏注意力融合kernel的实现难度**：需要CUDA编程经验。我们可以先尝试简单的triton实现，再优化为CUDA。
4. **提交包中的依赖管理**：确保所有依赖（如gptqmodel）都能在离线环境下安装（uv支持离线包吗？）。

你希望先从哪个讨论点开始？或者等实例恢复后，我们直接从第二阶段入手，边做边讨论？
