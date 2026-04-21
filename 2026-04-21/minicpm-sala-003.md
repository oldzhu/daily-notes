Compacted conversationLet me check the training script and understand the data requirements. 

Find EAGLE3 training data info

Searched for files matching `**/train_eagle3*`, no matches

Searched for files matching `**/eagle3*train*`, no matches

Searched for regex `train_eagle3|eagle3.*train|EAGLE3.*train|train.*eagle`, 5 results

Searched for files matching `**/eagle*train*`, no matches

Searched for regex `eagle.*draft.*train|draft.*model.*train|speculative.*train`, 2 results

Searched for files matching `**/eagle3*`, no matches

Searched for files matching `**/eagle*`, 21 matches

Searched for regex `EAGLE3|eagle3|Eagle3` (`**/docs/soar_2026_changes/**`), 19 results

Searched for regex `eagle3.*model|llama_eagle3|eagle.*config|EagleConfig`, 20 results

Searched for files matching `**/llama_eagle*`, 4 matches

Searched for regex `train.*draft|draft.*train|data.*eagle|eagle.*data|training.*spec`, 9 results



Fetched 2 resources



Fetched 2 resources



Fetched 2 resources

EAGLE3 training is a **knowledge distillation** process — the draft model learns to predict the target model's hidden states. Here's what's needed:

## Training Data for EAGLE3

### What the training data IS
NOT raw text. The training data consists of **pre-extracted hidden states** from the target model (MiniCPM-SALA BF16). Each sample is a `.pt` file containing:
- `hidden_state` — intermediate layer hidden states (the "aux" layers captured by `layers_to_capture`)
- `input_ids` — token IDs  
- `loss_mask` — valid token mask

### How it's generated (2-step process)

**Step 1: Data generation** — Run the BF16 target model on text inputs and save hidden states:
- The EAGLE repo has `eagle/train/ge_data/` scripts for this
- The recommended tool is now **[SpecForge](https://github.com/sgl-project/SpecForge)** (SGLang's official EAGLE training framework)
- Feed text through the target model → extract hidden states at specific layers → save as `.pt` files

**Step 2: Training** — Train the draft model's FC fusion + decoder layer to predict next-token hidden states using `SmoothL1Loss` (feature regression) + KL divergence (probability matching via the target's `lm_head`)

### What text data to feed in Step 1

This is the critical choice. Options:

| Source | Pros | Cons |
|--------|------|------|
| `perf_public_set.jsonl` (eval data, ~60 samples) | Perfect domain match (QA/CWE/MCQ/NIAH) | Too small — will overfit badly |
| ShareGPT (general conversations) | Large, proven for other EAGLE models | Domain mismatch with MiniCPM-SALA's tasks |
| Synthetic domain data | Can generate large amounts in-domain | Need to craft prompts for QA/CWE/MCQ/NIAH |
| Mix of eval + ShareGPT | Balance domain match + volume | Need curation |

### Practical recommendation

For MiniCPM-SALA, the ideal approach would be:
1. **Use the eval data** (`perf_public_set.jsonl`) to generate hidden states — but this alone is ~60 samples, far too few
2. **Supplement with ShareGPT** or similar conversational data to get thousands of samples
3. The EAGLE authors note: *"When Qwen2 is the target model... the training dataset is ShareGPT. If you want to use it on non-English data, please train with the corresponding data"* — **domain matching matters**

### Feasibility concerns for our competition

- **GPU time**: Training typically takes 1-2 days on 8× RTX 3090. We have 1× RTX 6000D and a 5-hour total budget (quantization + evaluation)
- **Storage**: Draft model is 612MB, counts toward the 2GB submission limit
- **Data availability**: We'd need to download or generate sufficient training data on fcloud
- **ROI question**: Even with a well-trained draft model, MiniCPM-SALA is a **small model** (~4B params) — the draft overhead (293M params forward + verify cost) may not yield a net speedup since the per-token latency of the target is already low

The biggest risk is that **the overhead of running the draft model may never pay off for a model this small**, regardless of training quality. EAGLE3 shines on large models (13B+) where each target token is expensive.
