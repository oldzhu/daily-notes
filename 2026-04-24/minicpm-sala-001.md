

Fetched https://mp.weixin.qq.com/s/SbzFY5KPpU3oQcqp8zT_Eg

I've read the article. Here's my analysis:

## 曹议's Logprob Method vs Our Task-Filter Method

### How the Logprob Method Works

1. **One-time baseline**: Run BF16 model on a set of prompts with greedy decoding, extract top-256 logprobs of the last 128 tokens, save as numpy files
2. **Per-config test**: Send same prompts to quantized model, compare logprob distributions via **KL divergence**, **cosine similarity**, and a third metric
3. Only requires prefill — no full generation needed. Zero GPU for baseline data.

### Comparison

| Dimension | Logprob Method (曹议's) | Task-Filter Method (ours) |
|-----------|------------------------|--------------------------|
| **Speed** | Very fast (seconds-minutes, prefill only) | 3-5 min (MCQ still generates ~6K tokens per sample) |
| **Stability** | Excellent — continuous metric, no sampling noise | Poor — 30 MCQ samples, each = 3.33%, high variance |
| **Measures what?** | Distribution closeness to BF16 (indirect proxy) | Actual end-to-end task accuracy (direct measure) |
| **Decode coverage** | Prefill only — **no decode error propagation** | Full generation — captures decode drift |
| **Sensitivity** | Can detect subtle quality differences between configs | Cannot distinguish configs within ~5% accuracy noise |
| **Predicts final score?** | No — low KL doesn't guarantee >97% accuracy | Yes — directly tells you accuracy |
| **Setup cost** | Needs one-time BF16 baseline run | Zero setup |
| **Infrastructure** | New script + baseline files | Just args to existing eval script |

### Key Insight

The two methods serve **different purposes**:

- **Logprob**: Best for **ranking** quantization configs relative to each other. If config A has KL=0.02 and config B has KL=0.05, A is unambiguously closer to BF16. Perfect for our Phase B where we need to compare B1/B2/B3/B4 and the MCQ variance makes 50% vs 56% indistinguishable.

- **Task-filter**: Best for **absolute validation** — does this config pass the 97%/99% threshold? You must eventually run the actual eval.

### The Critical Limitation 曹议 Acknowledges

> "只能感知 Prefill 阶段的精度误差，对 Decode 阶段的误差传播没有覆盖"

This matters a lot for our model: MCQ tasks generate ~6K output tokens with chain-of-thought. Quantization errors compound during autoregressive decode. A config might look great on prefill logprobs but produce degraded reasoning chains. This is likely why 曹议 still needs the actual eval for final validation.

### Verdict: Worth Implementing?

**Yes, worth it — but as a complement, not a replacement, and with appropriate priority.**

**Arguments for:**
- Our primary bottleneck for Phase B is re-quantization (85 min), not evaluation time. But if we do many rounds of tuning, the logprob method's stability advantage pays off — you can detect real differences instead of chasing MCQ noise
- Implementation is straightforward: ~100 lines using SGLang's `/v1/completions` with `logprobs=True`
- We have the BF16 model on fcloud (`/root/models/openbmb/MiniCPM-SALA-Copy`) for baseline

**Arguments against implementing now:**
- We only have 4 Phase B configs to test. The task-filter method is already working and "good enough" for 4 tests
- Each re-quantization takes 85 min — that dominates the pipeline, not the 3-5 min eval
- We'd need to design the prompt set carefully (should include long-context prompts to match eval distribution)

**My recommendation**: Proceed with Phase B using our current task-filter method first. If Phase B doesn't resolve the accuracy issue and we need many more iterations, implement the logprob tool at that point. The investment makes sense when we enter a tight tuning loop with >5-10 configs to compare.

Shall I proceed to commit and push the current changes for Phase B testing? 


