I'll help you analyze this competition and work towards winning it. Let me start by reading the competition page and gathering all the important information.
Now let me read the competition page:
Let me search for arxiv and github links in the page:
Let me check more pages from the competition site and the MiniCPM-SALA model page:
Let me search for paper link in the HuggingFace page:
Let me read the FAQ and toolkit pages, and also search for MiniCPM-SALA arxiv paper:
Now let me search for the MiniCPM-SALA arXiv paper and check the OpenBMB GitHub:
Now I have all the key information. Let me create a comprehensive competition project document:
## Summary: SOAR 2026 Competition - All Key Information

I've thoroughly reviewed the competition page and gathered all the essential information. Here's what you need to know:

---

### 🔗 Key Resources Found

| Resource | URL |
|----------|-----|
| **arXiv Paper** | https://arxiv.org/abs/2602.11761 |
| **GitHub (OpenBMB)** | https://github.com/OpenBMB |
| **Model (HuggingFace)** | https://huggingface.co/openbmb/MiniCPM-SALA |
| **Competition Site** | https://soar.openbmb.cn/competition |
| **Contact Email** | contact@openbmb.cn |

---

### 📄 Paper Details (arXiv:2602.11761)

**Title**: MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling

**Key Points**:
- **9B-parameter hybrid architecture** combining:
  - **Sparse attention (InfLLM-V2)** - for high-fidelity long-context modeling
  - **Linear attention (Lightning Attention)** - for global efficiency
- Uses **1:3 ratio** layer selection algorithm
- **Hybrid positional encoding (HyPE)**
- **75% reduction** in training costs vs. training from scratch
- **3.5x faster inference** at 256K tokens vs. full-attention model
- Supports up to **1M token context length** on single NVIDIA A6000D GPU

---

### 🎯 Competition Challenge

You must optimize inference performance for MiniCPM-SALA model on NVIDIA RTX PRO GPU:

**Optimization Focus Areas**:
1. Operator fusion (算子融合)
2. Kernel optimization
3. Memory & KV cache read/write optimization
4. Prefill/Decode path optimization
5. Graph compilation/operator tuning

**Allowed**: Model quantization (official quantized models provided)
**NOT Allowed**: Self-prepared models, prefix cache optimization

---

### 📊 Evaluation & Scoring

**Metric**: `benchmark_duration` (lower is better)

| Configuration | Weight |
|--------------|--------|
| `--max-concurrent 1` (no concurrency) | **40%** |
| `--max-concurrent 8` (low concurrency) | **30%** |
| No limit (high concurrency) | **30%** |

**Score Formula**:
```
S_N = (Duration_best / Duration_player) × 100
Performance Score = S₁ × 40% + S₈ × 30% + S_∞ × 30%
Final Score = Performance Score × Correctness Coefficient C
```

---

### 📅 Timeline

- **Feb 25, 2026 12:00 (UTC+8)**: Submission opens, complete rules released
- **May 27, 2026**: Registration deadline

---

### 🛠️ What I Need External Help With

As we prepare for this competition, here are areas where external assistance would be valuable:

1. **GPU Resources**: Access to NVIDIA RTX PRO GPU (or A6000/A100) for development and testing
2. **Large File Downloads**: The MiniCPM-SALA model is 9B parameters (~18GB+), requiring significant bandwidth
3. **CUDA Expertise**: Guidance on kernel optimization and memory access patterns
4. **SGLang Framework**: Deep understanding of the inference framework internals
5. **Profiling Tools**: Access to NVIDIA Nsight Compute/Systems for performance analysis

---

### 📁 Project Documentation

I've created a comprehensive project document saved at:
**`/home/z/my-project/download/SOAR2026_Competition_Project.md`**

This includes all the competition details, paper summary, optimization strategies, and technical areas to master.

---

### Next Steps (If You Want to Proceed)

1. **Study the paper in depth** - Understand InfLLM-V2 and Lightning Attention mechanisms
2. **Set up development environment** - Get access to RTX PRO GPU
3. **Download and profile MiniCPM-SALA** - Understand the baseline performance
4. **Master SGLang** - Learn the inference framework and optimization knobs
5. **Develop optimization strategy** - Focus on high-impact areas (kernel fusion, KV cache)

Would you like me to proceed with any specific aspect of the competition preparation, such as analyzing the SGLang framework or creating a detailed optimization plan?
