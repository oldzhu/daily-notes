Let me check the correct way to expose files through the web server:
The files are on the server. Let me create a proper download API endpoint that you can access:
Let me test if the API endpoint works:
The files are accessible! Let me check what URL format works for you:
Let me try a different approach - I'll base64 encode the files so you can download them:
Since the direct file links aren't working, let me provide the full content here in the chat. I'll show both documents:

---

# 📄 Document 1: MiniCPM-SALA Paper Study

---

# MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling

**Paper Deep Dive & Implementation Guide | arXiv:2602.11761 | SOAR 2026 Competition Reference**

---

## 1. Introduction and Overview

MiniCPM-SALA represents a groundbreaking approach to solving one of the most critical challenges in modern Large Language Models (LLMs): the efficient processing of ultra-long contexts. The evolution of LLMs has shifted from simple question-answering to complex applications requiring deep understanding of entire technical manuals, repository-scale code engineering, and long-horizon agent tasks. This paradigm shift demands models capable of handling millions of tokens, a scale where traditional Transformer architectures face severe computational and memory bottlenecks.

The core innovation of MiniCPM-SALA lies in its hybrid architecture that strategically combines two complementary attention mechanisms: sparse attention (InfLLM-V2) for high-fidelity long-context modeling, and linear attention (Lightning Attention) for global computational efficiency. This hybrid design, implemented with a 1:3 ratio (25% sparse, 75% linear), achieves the best of both worlds—maintaining the precision necessary for complex reasoning tasks while dramatically reducing the quadratic complexity that plagues traditional full-attention models.

### 1.1 Key Contributions

- **Hybrid Architecture**: Integration of 25% InfLLM-V2 (sparse) and 75% Lightning Attention (linear) for balanced throughput and precision
- **Efficient Training**: Transformer-to-hybrid conversion reduces training costs by ~75% compared to training from scratch
- **HyPE Positional Encoding**: Hybrid positional encoding for harmonized short and long context performance
- **Performance**: 3.5x faster inference at 256K tokens on NVIDIA A6000D; supports up to 1M token context on single GPU

---

## 2. Model Architecture Overview

MiniCPM-SALA adopts a hybrid architecture that interleaves sparse attention layers and linear attention layers while retaining the Feed-Forward Network (FFN) block after each attention block to ensure high-capacity knowledge representation. The architectural design is inspired by recent representative studies such as Qwen3-Next and Kimi-Linear, employing a 1:3 mixing ratio where 25% of layers adopt sparse attention while the remaining 75% employ linear attention. This configuration leverages the complementary strengths of both attention mechanisms—linear attention layers provide constant computational and memory complexities, while sparse attention layers facilitate effective modeling of long-range dependencies.

| Component | Percentage | Complexity | Purpose |
|-----------|------------|------------|---------|
| Sparse Attention (InfLLM-V2) | 25% | O(N × K) | High-fidelity long-context |
| Linear Attention (Lightning) | 75% | O(N) | Global efficiency |

---

## 3. Sparse Attention: InfLLM-V2

InfLLM-V2 serves as the sparse attention mechanism in MiniCPM-SALA, offering the distinct advantage of introducing no additional parameters to the architecture. Its inherent flexibility allows seamless switching between dense and sparse modes, which is highly compatible with the conversion process from standard Transformers. This compatibility facilitates stable training initialization by allowing sparse modules to inherit dense weights without architectural discrepancies.

### 3.1 Core Mechanism

The InfLLM-V2 approach organizes the KV-Cache into a two-level structure: a local sliding window for nearby context and a global block-memory for long-range dependencies. The key innovation lies in its block-level attention scoring mechanism, where instead of computing full attention over all positions, it first identifies relevant blocks through a compressed representation and then performs detailed attention only within those selected blocks.

**Key Formula: Block Selection**

Given query Q and compressed key K', the block attention score is computed as:

```
Score(Q, K') = Softmax(Q × K'ᵀ / √d)
```

Top-k blocks are selected based on this score for detailed attention computation.

**Implementation Code: Block Compression and Selection**

```python
# Key compression using mean pooling over blocks
compressed_k = filtered_k.mean(dim=1)  # Shape: [num_blocks, heads, dim]

# Block selection via max pooling over attention scores
block_score = max_pooling_1d_varlen(score, cu_seqlens_q, cu_seqlens_k,
                                    cache_lens, block_size, stride)
topk_idx = block_score.topk(topk, dim=-1).indices
```

**Configuration Parameters**

The sparse attention mechanism is configured with the following parameters from the model config. These parameters control the granularity of block selection and the trade-off between computational efficiency and precision in long-context scenarios. The kernel_size of 32 determines the block granularity, while topk=64 limits the number of blocks selected for detailed attention computation, ensuring bounded memory usage regardless of sequence length.

```json
"sparse_config": {
    "kernel_size": 32,      // Block size for key compression
    "kernel_stride": 16,    // Stride for sliding window over blocks
    "block_size": 64,       // Block size for sparse attention
    "window_size": 2048,    // Local attention window
    "topk": 64,             // Number of blocks to attend to
    "dense_len": 8192       // Initial dense attention length
}
```

---

## 4. Linear Attention: Lightning Attention

Lightning Attention provides the linear attention component in MiniCPM-SALA, selected for its functional proximity to standard softmax attention. This structural alignment mitigates the complexities of parameter adaptation during the Transformer-to-hybrid conversion, preserving pre-trained knowledge and ensuring robust downstream performance. Lightning Attention also provides better length generalization capabilities, improving data efficiency during long-context continual-training.

### 4.1 Mathematical Foundation

The key insight of linear attention is to reformulate the attention computation using kernel-based feature maps, enabling the associative property of matrix multiplication to reduce complexity from O(N²) to O(N). Instead of computing the attention matrix explicitly, linear attention computes a recurrent state that accumulates information across the sequence, making inference time constant regardless of sequence length.

**Standard vs Linear Attention**

```
Standard Attention: O = Softmax(Q × Kᵀ / √d) × V    Complexity: O(N²)

Linear Attention: O = φ(Q) × [φ(K)ᵀ × V]            Complexity: O(N × d²)
```

Where φ(·) is a kernel feature map (typically ELU+1 or similar non-linear transformation). The bracketed term φ(K)ᵀ × V can be computed incrementally as a recurrent state, enabling constant memory inference.

**Implementation: Lightning Attention Core**

```python
# Lightning Attention uses chunk_simple_gla from fla library
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla

# Recurrent state update (decoding phase):
state['recurrent_state'] = k_t.T @ v_t  # Incremental state update
output = q_t @ state['recurrent_state'] / scale
```

**Cache Implementation**

The LightningCacheLayer maintains a compact recurrent state instead of storing all past key-value pairs. This is the key to achieving constant memory usage during generation. The state dictionary stores only the accumulated key-value product and other necessary intermediate values, enabling efficient inference for extremely long sequences without the memory overhead of traditional KV-Cache.

```python
class LightningCacheLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.state = {}  # Stores recurrent_state, conv_state, ffn_state
    
    def update(self, recurrent_state=None, attn_state=None, ...):
        if recurrent_state is not None:
            self.state['recurrent_state'] = recurrent_state
```

---

## 5. Hybrid Positional Encoding (HyPE)

HyPE (Hybrid Positional Encoding) represents a crucial architectural innovation that balances rich positional awareness with long-range information retention. The key insight is that different attention mechanisms have different requirements for positional information—linear attention benefits from explicit positional encoding to maintain token order awareness, while sparse attention can operate more effectively without positional constraints that might limit its ability to retrieve distant information.

### 5.1 Design Rationale

- **Linear Attention Layers**: Apply RoPE (Rotary Positional Embedding) to facilitate position-sensitive memory, preserving relative order of tokens within global context
- **Sparse Attention Layers**: Remove RoPE to prevent decay of long-distance information, enabling more precise recall over extended contexts

**Configuration in Model**

```python
# From config.json:
"attn_use_rope": false,        # Sparse attention: NO RoPE
"lightning_use_rope": true,    # Linear attention: USE RoPE
```

---

## 6. Layer Selection Algorithm

Rather than uniformly interleaving sparse and linear attention layers, MiniCPM-SALA employs a sophisticated layer selection mechanism to determine the optimal placement of sparse attention modules. This approach, based on HALO (Hybrid Attention via Layer Optimization), results in superior downstream performance compared to naive uniform interleaving.

### 6.1 Layer Type Distribution

The model config reveals a carefully designed layer distribution pattern across the 32 transformer layers. The mixer_types array specifies which attention mechanism each layer uses. Notice that the first and last layers are always sparse attention ('minicpm4') for training stability, while the middle layers follow a strategic pattern optimized by the HALO algorithm.

```json
"mixer_types": [
    "minicpm4",           // Layer 0: Sparse (always first)
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "minicpm4",           // Layer 9: Sparse (strategic placement)
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "lightning-attn", "lightning-attn",
    "minicpm4", "minicpm4",   // Layers 16-17: Sparse cluster
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "minicpm4",           // Layer 22: Sparse
    "lightning-attn", "lightning-attn", "lightning-attn", "lightning-attn",
    "lightning-attn", "lightning-attn",
    "minicpm4", "minicpm4", "minicpm4"   // Layers 29-31: Sparse (last)
]
```

The distribution shows 8 sparse attention layers ('minicpm4') and 24 linear attention layers ('lightning-attn'), achieving the stated 1:3 ratio. The strategic clustering of sparse layers near the end of the network is particularly interesting, as this is where higher-level semantic understanding becomes crucial for final token predictions.

---

## 7. Training Methodology (HALO Framework)

MiniCPM-SALA employs a sophisticated multi-stage training process starting from an intermediate checkpoint of MiniCPM-4.0, which had already been trained on 7T tokens. This Transformer-to-hybrid conversion approach, based on the HALO framework, reduces total training costs by approximately 75% compared to training from scratch, requiring only ~2T tokens versus the 8T tokens needed for the original model.

### 7.1 Training Stages

| Stage | Trainable Params | Sparse | Seq Len | Tokens |
|-------|-----------------|--------|---------|--------|
| Architecture Conv. | Linear Attn | Off | 0.5K | 1.3B |
| Continual Stable | All | Off | 4K | 314.6B |
| Short-Decay | All | Off | 4K | 1T |
| Long-Decay | All | On | 32K→520K | 215.7B |
| SFT | All | On | 64K→140K | 417.8B |

---

## 8. Additional Architectural Improvements

### 8.1 QK-Normalization

Applied to all attention layers (both sparse and linear), QK-Normalization prevents activation spikes that often occur in long-context training and improves the expressivity of linear attention modules. This technique normalizes the query and key vectors before computing attention scores, ensuring numerical stability during training on extremely long sequences.

```python
# Config: "qk_norm": true
```

### 8.2 Output Gates

Output gates are incorporated after each attention block to mitigate attention sink issues. By regulating information flow, the output gate prevents excessive focus on specific tokens and ensures more flexible distribution of attention weights. This aligns with recent advances in gated attention mechanisms.

```python
# Config: "use_output_gate": true, "attn_use_output_gate": true
```

---

## 9. Key Insights for SOAR Competition

Understanding the MiniCPM-SALA architecture is crucial for optimizing inference performance in the SOAR competition. The hybrid design offers multiple optimization opportunities across different components, from kernel-level optimizations for both attention mechanisms to memory management strategies for the hybrid cache system.

### 9.1 Optimization Opportunities

- **Operator Fusion**: Combine the block selection and attention computation in sparse attention layers
- **Kernel Optimization**: Optimize the chunk_simple_gla and fused_recurrent_simple_gla kernels for linear attention
- **KV Cache Management**: Efficient handling of the hybrid cache system (InfLLMv2CacheLayer + LightningCacheLayer)
- **Prefill/Decode Paths**: Different optimization strategies for prefill (chunked parallel) vs decode (recurrent) phases
- **SGLang Integration**: Leverage SGLang's inference framework capabilities for the hybrid architecture

### 9.2 Key Parameters to Optimize

| Parameter | Value | Impact |
|-----------|-------|--------|
| hidden_size | 4096 | Model capacity |
| num_hidden_layers | 32 | Depth |
| num_attention_heads | 32 | Attention parallelism |
| num_key_value_heads | 2 (GQA) | KV cache efficiency |
| max_position_embeddings | 524288 (512K) | Max context length |
| head_dim | 128 | Per-head dimension |

---

## 10. Summary

MiniCPM-SALA demonstrates that hybrid sparse-linear attention architectures can achieve performance comparable to full-attention models while offering dramatic efficiency improvements. The key innovations include the strategic 1:3 layer distribution, HyPE for positional encoding, QK-normalization for stability, and output gates for better attention distribution. For the SOAR competition, optimization efforts should focus on kernel-level improvements for both attention mechanisms, efficient cache management, and leveraging the distinct characteristics of prefill versus decode phases.

---

# 📄 Document 2: SOAR 2026 GPU Cloud Comparison

---

# SOAR 2026 Competition - Cloud GPU Instance Comparison Guide

**RTX PRO GPU Access for MiniCPM-SALA Optimization**

*Including fcloud (Competition Sponsor) & Major Cloud Providers*

*February 2026*

---

## 1. Competition Hardware Requirements

The SOAR 2026 competition officially states: "本次比赛官方将采用 NVIDIA 高端 RTX PRO GPU 进行评测。选手需自行准备或租用 NVIDIA 高端 RTX PRO GPU（或等效）资源进行开发与测试。" This means participants need to prepare or rent NVIDIA high-end RTX PRO GPU (or equivalent) resources for development and testing.

### 1.1 NVIDIA RTX PRO 6000 Blackwell Server Edition

The RTX PRO 6000 is NVIDIA's flagship Blackwell-based workstation GPU, specifically designed for AI workloads with exceptional memory capacity and bandwidth.

| Specification | Value |
|--------------|-------|
| Architecture | NVIDIA Blackwell |
| GPU Memory | **96 GB GDDR7 with ECC** |
| Memory Bandwidth | 1,792 GB/s |
| CUDA Cores | 24,064 |
| Tensor Cores | 752 |
| Ray Tracing Cores | 188 |
| Peak FP4 AI Performance | 4 PFLOPS |

---

## 2. fcloud (Competition Sponsor)

fcloud (fcloud.cn) is one of the sponsors of the SOAR 2026 competition. As a Chinese cloud computing provider, they specialize in AI computing infrastructure and GPU cloud services. Their platform offers GPU Cloud Service, Bare Metal Service, and various AI LLM platforms including Model API Service and AI TIC Platform.

### 2.1 fcloud Overview

- **Website**: https://www.cnfcloud.com/
- **Services**: GPU Cloud Service, Bare Metal Service, Model API Service
- **Focus**: AI computing, LLM training, high-performance computing
- **Key Feature**: "FCloud 2.0 Major Upgrade: Launch of China's First Batch of New High-Performance Computing Cards"
- **Contact Hotline**: 13361960328 (10:00-19:00)

### 2.2 Pros and Cons

**Advantages:**
- Official competition sponsor - likely optimized for MiniCPM-SALA
- Chinese language support and local customer service
- Potential for competition-specific discounts or support
- Direct integration with OpenBMB ecosystem

**Considerations:**
- Specific pricing not publicly listed - need to contact for quotes
- Pricing may vary based on demand and availability
- Account registration and verification required

---

## 3. International Cloud GPU Providers

### 3.1 RunPod

RunPod is a leading GPU cloud platform that enables developers, researchers, and AI companies to deploy and scale GPU workloads. They offer competitive pricing and excellent availability for RTX PRO GPUs.

| GPU Model | Price/Hour | Notes |
|-----------|------------|-------|
| **RTX PRO 6000** | **$1.89/hr** | Competition GPU |
| H100 SXM | $1.99-$2.99/hr | Data center grade |
| A100 80GB | ~$1.50/hr | 80GB VRAM |
| RTX 4090 | ~$0.40-$0.70/hr | Budget option |

### 3.2 Vast.ai

Vast.ai operates a GPU marketplace model where individual providers offer compute resources. This often results in the lowest prices but availability can vary.

**RTX PRO 6000 S: $0.93/hr** (lowest price found)

### 3.3 Major Cloud Providers (AWS, GCP, Azure)

Hyperscaler cloud providers offer enterprise-grade reliability but at premium prices. RTX PRO GPUs may have limited availability on these platforms as they primarily offer data center GPUs.

| Provider | GPU | Price/Hour | Notes |
|----------|-----|------------|-------|
| AWS | A100 80GB | ~$3.06/hr | P5 instance |
| GCP | H100 | ~$3.50/hr | A3 instance |
| Azure | A100 80GB | ~$3.06/hr | ND-series |

---

## 4. Chinese Cloud GPU Providers

Chinese cloud GPU providers offer competitive pricing, especially for RTX 4090 and other consumer GPUs. Pricing is typically quoted in RMB (CNY). These are useful for development and testing, but note that the competition requires RTX PRO GPUs for final submission.

| Provider | RTX 4090 | A100 | RTX PRO | Notes |
|----------|----------|------|---------|-------|
| **智星云** | ¥1.35/hr | - | TBD | Best value |
| BuluAI | ¥1.93/hr | - | TBD | Good option |
| 丹摩DAMODEL | ¥2.18/hr | - | TBD | |
| AutoDL | ¥2.19/hr | - | TBD | Popular |
| 恒源云 | ¥2.0-3.22/hr | - | TBD | |
| 阿里云 | Variable | Available | Contact | Enterprise |

---

## 5. Comprehensive Comparison

### 5.1 RTX PRO 6000 Pricing Comparison

| Provider | Price/Hour | Competition Fit | Recommendation |
|----------|------------|-----------------|----------------|
| **Vast.ai (Spot)** | **$0.93/hr** | ✓ Exact GPU | **Best Price** |
| **RunPod** | **$1.89/hr** | ✓ Exact GPU | **Best Reliability** |
| **fcloud (Sponsor)** | Contact for quote | ✓ Official Support | Official Sponsor |

### 5.2 Provider Pros and Cons Summary

**Vast.ai**

**Pros:** Lowest prices ($0.93/hr for RTX PRO 6000), flexible marketplace model, good for development/testing

**Cons:** Spot instances may be preempted, variable availability, less enterprise support

**RunPod**

**Pros:** Reliable platform ($1.89/hr), pay by millisecond, instant deployment, HIPAA & GDPR compliant, good documentation

**Cons:** Higher price than Vast.ai, credit card required for sign-up

**fcloud (Competition Sponsor)**

**Pros:** Official competition sponsor, local Chinese support, potential for competition-specific offers, OpenBMB ecosystem integration

**Cons:** Pricing not publicly listed, may require verification process

---

## 6. Recommendations

### 6.1 Development Phase

For development and initial testing, consider using cheaper GPUs like RTX 4090 or A100. Chinese providers like 智星云 (¥1.35/hr for RTX 4090) offer excellent value for algorithm development and debugging. This allows you to iterate quickly on your optimization approach before testing on the competition hardware.

### 6.2 Final Testing Phase

For final testing and submission preparation, use RTX PRO 6000 to ensure your solution is optimized for the competition hardware. RunPod offers the best balance of reliability and price ($1.89/hr). Vast.ai provides the lowest price ($0.93/hr) but with less reliability. Contact fcloud for potential sponsor discounts.

### 6.3 Cost Estimation

Assuming 100 hours of development on RTX 4090 + 20 hours of final testing on RTX PRO 6000:

- Development (智星云 RTX 4090): ¥135 (~$18.50)
- Testing (RunPod RTX PRO 6000): $37.80
- **Total Estimated Cost: ~$56**

---

## 7. Quick Start Guide

### 7.1 RunPod Setup

1. Visit https://www.runpod.io/ and create an account
2. Add payment method (credit card required)
3. Navigate to GPU Cloud > RTX PRO 6000
4. Select template with PyTorch and SGLang pre-installed
5. Deploy and start testing MiniCPM-SALA

### 7.2 fcloud Setup

1. Visit https://www.cnfcloud.com/ and register
2. Contact support (13361960328) for RTX PRO GPU availability
3. Mention SOAR 2026 competition participation
4. Request pricing for RTX PRO GPU instance

---

## 8. Important Notes

- Prices are subject to change and may vary by region and availability
- The competition evaluation uses RTX PRO GPU - ensure your final testing is on equivalent hardware
- Spot/preemptible instances may not be suitable for long training runs
- Check the official competition page (https://soar.openbmb.cn) for any hardware updates
- Contact competition organizers at contact@openbmb.cn for clarifications

---

## 9. Summary

For the SOAR 2026 competition requiring RTX PRO GPU access, participants have several viable options. RunPod offers the best balance of reliability and price at $1.89/hr for the RTX PRO 6000. Vast.ai provides the lowest price at $0.93/hr but with less guaranteed availability. As an official sponsor, fcloud should be contacted for potential competition-specific offers and support. For development, Chinese providers like 智星云 offer excellent value for RTX 4090 instances at around ¥1.35/hr.

| Use Case | Recommended | GPU | Est. Cost |
|----------|-------------|-----|-----------|
| Development | **智星云 / AutoDL** | RTX 4090 | ~¥1.35-2.2/hr |
| Final Testing | **RunPod** | RTX PRO 6000 | $1.89/hr |
| Budget Option | Vast.ai (Spot) | RTX PRO 6000 | **$0.93/hr** |
| Official Support | fcloud (Sponsor) | Contact for info | TBD |

---

**That's all the content from both documents!** You can copy and save this as needed.
