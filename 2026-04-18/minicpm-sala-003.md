Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/globalStorage/github.copilot-chat/memory-tool/memories/soar_2026_leaderboard.md)

Updated memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/champion_blog_analysis.md)

This is a **major strategic update**. Let me analyze the impact:

---

## Analysis of the New Speed Dataset

The new dataset is **heavily long-context** — 68% of inputs are 32K-512K tokens. This fundamentally changes optimization priorities:

| Phase | Old Dataset (short) | New Dataset (long-context) |
|-------|-------------------|--------------------------|
| **Prefill** | Minor fraction of time | **Dominates latency** (512K inputs!) |
| **Decode** | Major fraction of time | Smaller fraction (58% outputs < 512 tokens) |

### Impact on Our Two Tracks

**NVFP4 (importance: ↑↑ INCREASED)**
- FP4×FP4 native tensor cores accelerate **compute** — directly helps prefill for long sequences
- Long prefill is compute-bound → FP4 GEMM throughput gain is maximized
- This may now be MORE valuable than EAGLE3 short-term

**EAGLE3 (importance: ↓ relatively decreased, but still valuable)**
- Spec decode accelerates **decode only** — helps less when prefill dominates
- BUT: 18% of outputs are 4K-32K tokens — spec decode still matters for those
- Long-term still the right direction, just smaller relative impact on new dataset

### Recommended Testing Strategy

**Test them one-by-one, sequentially**, for these reasons:
1. **Isolation**: We need to know each one's standalone impact to make good decisions
2. **Debugging**: If something fails (accuracy drop, server crash on 512K context), we need to know which change caused it
3. **Risk**: NVFP4 has accuracy risk; EAGLE3 has functional risk on long-context SimpleGLA state management — testing both simultaneously makes debugging impossible
4. **Resource**: Both need the BF16 model loaded for quantization/training — GPU memory constraint

**Recommended order**:
1. **EAGLE3 training first** — it only uses the BF16 model for forward passes, doesn't change the serving model. Run training, save draft model.
2. **NVFP4 quantization second** — also uses BF16 model. Run quantization, save NVFP4 checkpoint.
3. **Test EAGLE3 with GPTQ baseline** — measure acceptance rate and speed with current config
4. **Test NVFP4 standalone** — measure accuracy and speed
5. **If both work: combine** — NVFP4 target + EAGLE3 draft for maximum throughput

This way training/quantization runs in parallel conceptually (both can be done in one fcloud session), then we test serving sequentially.

**However**, given the new long-context dataset, I'd suggest **prioritizing NVFP4 testing** for the next submission since it directly helps prefill. EAGLE3 training can run first (it's a one-time cost), but the first submission should test NVFP4 serving.

Shall I start the EAGLE3 training on fcloud as soon as the instance is ready? The training will take ~30-60 minutes. While it runs, I can prepare the NVFP4 test environment. After training completes, we can run NVFP4 quantization, then test both serving configs sequentially.
