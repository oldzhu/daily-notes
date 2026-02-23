I'll continue working on the in-depth paper study and create the comprehensive document. Let me first check what files we have and then gather the implementation code from HuggingFace to create the detailed paper explanation.
Let me get the full HTML version of the paper and the model code from HuggingFace to create the comprehensive document.
Now let me use the CLI to read the full HTML version of the paper and get the HuggingFace model code.
Let me search for specific formulas and architecture details in the paper HTML.
Now let me get the model implementation code from HuggingFace:
Now let me get the actual model implementation code:
Now I have all the information needed. Let me create the comprehensive paper study document with formulas, algorithms, and code mapping.
Now let me create the comprehensive paper study document:
I've successfully created a comprehensive paper study document for MiniCPM-SALA. Here's what was accomplished:

## Document Created

**File:** `/home/z/my-project/download/MiniCPM_SALA_Paper_Study.docx`

## Document Contents

The document provides an in-depth, intuitive explanation of the MiniCPM-SALA paper with the following sections:

### 1. Introduction and Overview
- Problem statement: Ultra-long context processing in LLMs
- Key innovation: Hybrid sparse + linear attention architecture
- Performance highlights: 3.5x faster at 256K tokens, supports 1M context

### 2. Model Architecture Overview
- **1:3 Ratio**: 25% sparse attention (InfLLM-V2) + 75% linear attention (Lightning)
- Layer distribution table with complexity analysis

### 3. Sparse Attention (InfLLM-V2)
- Core mechanism explained with formulas
- Block selection formula: `Score(Q, K') = Softmax(Q × K'ᵀ / √d)`
- Implementation code from `modeling_minicpm_sala.py`
- Configuration parameters: `kernel_size=32`, `block_size=64`, `topk=64`, `window_size=2048`

### 4. Linear Attention (Lightning Attention)
- Mathematical foundation comparing O(N²) vs O(N) complexity
- Kernel-based feature map formulation
- Cache implementation code for recurrent state
- Key insight: `state['recurrent_state'] = k_t.T @ v_t`

### 5. HyPE (Hybrid Positional Encoding)
- Linear attention layers use RoPE
- Sparse attention layers remove RoPE
- Configuration: `attn_use_rope: false`, `lightning_use_rope: true`

### 6. Layer Selection Algorithm
- Complete `mixer_types` array showing all 32 layers
- Strategic placement: sparse at beginning, end, and key middle positions
- Pattern explanation for HALO optimization

### 7. Training Methodology (HALO Framework)
- 5-stage training pipeline table
- ~2T tokens (25% of training from scratch)
- Progressive context length: 0.5K → 4K → 32K → 160K → 520K

### 8. Additional Architectural Improvements
- QK-Normalization for stability
- Output Gates for attention sink mitigation

### 9. Key Insights for SOAR Competition
- Optimization opportunities (operator fusion, kernel optimization, KV cache management)
- Key model parameters table
- Prefill vs decode optimization strategies

### 10. Summary

## Key Resources Found

| Resource | URL |
|----------|-----|
| arXiv Paper | https://arxiv.org/abs/2602.11761 |
| HuggingFace Model | https://huggingface.co/openbmb/MiniCPM-SALA |
| Competition Page | https://soar.openbmb.cn/competition |

## Key Technical Details for Optimization

1. **Model Size**: 9B parameters, 32 layers, 4096 hidden dimension
2. **Attention**: 32 heads with GQA (2 KV heads)
3. **Max Context**: 524,288 (512K) tokens
4. **Key Libraries**: `infllm_v2`, `fla.ops.simple_gla`

The document combines paper formulas with actual implementation code from the HuggingFace model files, making it easy to understand both the theory and practice. This should serve as an excellent reference for your SOAR competition optimization work!
