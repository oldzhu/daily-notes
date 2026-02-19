I've reviewed the competition page and gathered the key information you need. The paper and code are indeed available, and I've outlined a structured preparation roadmap below to help you get started.

### 📄 Key Resources: Paper & Code
Here are the direct links to the foundational materials for the competition.

*   **arXiv Paper:** The technical report, titled "**MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling**," is available on arXiv at this link: [https://arxiv.org/abs/2602.11761](https://arxiv.org/abs/2602.11761) .
*   **GitHub Repository:** The official code and model are hosted in the `OpenBMB/MiniCPM` repository. You can find the specific documentation for the SALA architecture here: [https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf) .
*   **Model on Hugging Face:** The MiniCPM-SALA model itself is also available on Hugging Face for direct download and testing: [https://huggingface.co/openbmb/MiniCPM-SALA](https://huggingface.co/openbmb/MiniCPM-SALA) .

### 🧠 Understanding the Core Technology (MiniCPM-SALA)
To optimize effectively, you need to understand what you're optimizing. The competition centers on the **MiniCPM-SALA model**, a 9B-parameter model with a novel hybrid architecture . Here are its key features:

*   **Hybrid Architecture (SALA)**: It combines two types of attention layers to break the efficiency ceiling of traditional models .
    *   **Sparse Attention (InfLLM v2)** : 25% of the layers. These are "precise" and focus on retrieving critical information from the very long context.
    *   **Linear Attention (Lightning Attention)** : 75% of the layers. These are "fast" and handle the global context efficiently.
*   **HyPE Position Encoding**: A clever mix that uses RoPE for linear layers (to retain short-context performance) and NoPE for sparse layers (to avoid long-distance decay), enabling smooth scaling to millions of tokens .
*   **The Goal**: This architecture is designed to push the limits of long-context inference on a single consumer-grade GPU (like the NVIDIA RTX PRO series specified in the competition) . Your task is to make it run even faster.

### 🗺️ Your Roadmap to Prepare and Compete
Winning a competition like this requires a systematic approach. Here is a step-by-step guide to get you from zero to contender.

**Phase 1: Foundational Knowledge & Setup**
1.  **Deep Dive into the Paper and Code**: Start by thoroughly reading the arXiv paper  to understand the model's theory. Then, explore the GitHub repository to see how it's implemented. Pay close attention to the model definition and any existing kernels.
2.  **Master the Tools**: The competition explicitly names the optimization targets (operator fusion, kernel optimization, etc.) and the framework (**SGLang**) . You must become proficient in:
    *   **SGLang**: Study its performance optimization guides and server arguments. Key concepts include understanding its **attention backends** (like FlashAttention) and advanced features like **Prefill-Decode Disaggregation** .
    *   **NVIDIA Tools**: Since the evaluation is on NVIDIA RTX PRO GPUs, knowledge of profiling tools like **Nsight Systems** and **PyTorch Profiler** is non-negotiable for identifying bottlenecks .
    *   **Kernel Programming (Optional but High-Impact)** : For the most extreme optimizations, familiarity with **CUDA** and libraries like **CUTLASS** (for custom kernel fusion) would be a massive advantage. The competition's focus on "sparse operator fusion" points directly to this level of work .

**Phase 2: Analysis & Strategy (The 40/30/30 Rule)**
The scoring formula is your strategic guide: `Total Score = (S₁ × 40%) + (S₈ × 30%) + (S_∞ × 30%) × Correctness` .
1.  **Focus on the Heavy Weight**: The **single-request scenario (40% weight)** is your top priority. Optimizations here directly reduce the end-to-end latency for a single, long-context task. This is where techniques like **kernel fusion** (combining multiple operations into one kernel) and **memory access optimizations** will shine, as they reduce overhead on a single stream of work .
2.  **Don't Ignore Concurrency**: The low-concurrency (8 requests) and high-concurrency scenarios account for 60% of the performance score combined. Your single-request optimizations must play nicely with batching and parallel processing. Strategies like **Prefill-Decode (PD) Disaggregation** become highly relevant here, as they can prevent the compute-bound prefill phase from blocking the memory-bound decode phases of concurrent requests .
3.  **Correctness is King**: The final score is multiplied by a correctness coefficient `C` . A super-fast but slightly wrong model will score a zero. Always validate your optimization's output against a known-good baseline.

**Phase 3: Experimentation & Iteration**
1.  **Start Simple, Measure Everything**: Begin by running the model with the default SGLang configuration on your hardware. Establish a baseline performance for all three concurrency levels.
2.  **Iterate on Low-Hanging Fruit**: Experiment with different SGLang attention backends, enable `torch.compile`, and try different quantization methods offered by the framework. Measure the impact of each change meticulously.
3.  **Profile to Find Bottlenecks**: Use Nsight Systems to profile your best-performing configuration. Is the model kernel-bound, memory-bound, or perhaps spending too much time on data movement between the prefill and decode phases? Let the profiler guide your next, more advanced steps .
4.  **Go for the Advanced Techniques**: Based on your profiling, decide on your major optimization. This could involve:
    *   Implementing a custom fused kernel for a specific part of the SALA attention mechanism using CUTLASS .
    *   Implementing and tuning PD Disaggregation to better handle the mixed concurrency workloads .
    *   Exploring speculative decoding techniques supported by SGLang to speed up the decode phase .
5.  **Document and Package**: The competition requires code to be open-sourced with clear documentation for reproducibility . Keep a detailed log of your experiments and ensure your final submission includes a clean script to replicate your environment and run your optimized solution.

You need to set up a development environment with an NVIDIA RTX PRO GPU (or an equivalent like an A6000D, as mentioned in the paper ) and get the baseline model running through SGLang. Which part of the setup or the initial benchmarking would you like to dive into first?
