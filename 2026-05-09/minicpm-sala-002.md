pls document the above discussion if u haven't. let us have more disciussion, I will start fcloud so u can inspect MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 config.json to check if the quanted model has ny lighting entries in mixer_types, I think normally it will has 8 sparse attention layer and 24 lighting attention layers. but we'd btter to confirm. As almost all champions posts using w4a16+mixed fp4/fp8 kv cache(they didn't mention force dense or using sparse default - but I guess they use sparse  otherwise they would mention? so I guess w4a16+mixed fp4/fp8 kv cache+sparse can get a good accuracy ) , so I want to retest the combination w4a16+fp8 kv cache+sparse again, another reason is that test8b shows duration_total is 2411s which is quite short may means better speed and throughts?
======
Ran terminal command:  cat >> /home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md <<'EOF'

---

## Round 13d — pre-decision discussion: best sparse results + per-layer behavior of `--force-dense-minicpm`

### User asks
1. What are our best accuracy + benchmark results for `W4A16 + FP8 KV + sparse` and `W4A16 + bf16 KV + sparse`?
2. Under `--force-dense-minicpm`, sparse-attention layers run as dense softmax (no top-k, no sparse ops) — confirmed. What about lightning layers? Do they also become dense?

### Agent answer

**Q1 — best results we have**

| Config | Best test | ori_acc | norm | C | mcq / qa / niah / cwe / fwe | accuracy-eval duration | Notes |
|---|---|---|---|---|---|---|---|
| GPTQ + FP8 KV + **sparse** | Test 8b (`9d3ecd168`) | 76.07% | 95.08% | **0** | 60 / 66.67 / 96.67 / 60.33 / 96.67 | 2411 s | FP8 KV breaks sparse top-k scoring (descale rounding) |
| GPTQ + bf16 KV + **sparse** | Test 9 (`79e49f39f`) | **79.67%** | **99.58%** | **1.0** | 63.33 / 56.67 / 100 / 81.67 / 96.67 | 3157 s | Only sparse config that reaches C=1.0 |
| GPTQ + FP8 KV + **dense** (current baseline) | Test 12 (`9e82efe43`) | 79.29% | 99.11% | **1.0** | 63.33 / 72 / 97.78 / 100 / 63.33 | 4244 s | S1=121.71s, S8=44.09s, Smax=35.86s |

Key observation: **neither sparse config has S1/S8/Smax measured separately**. The "duration" column is the accuracy-eval total, not the speed-benchmark trio. Test 8b's 2411 s being smaller than Test 12's 4244 s is the duration of *running 150 accuracy samples at concurrency 8*, not S1/S8/Smax. We cannot conclude "sparse is faster" from it without dedicated speed runs.

User raised the point that Test 8b's short eval duration could hint at better throughput. Counterpoint: in dense FP8 we already serve >100 tok/s, so a 2411s/4244s ratio is dominated by output-length differences (sparse runs may have shorter outputs because of accuracy regression — when answers are wrong/truncated they finish sooner, NOT because the model decoded faster).

**Q2 — per-layer behavior under `--force-dense-minicpm`**

Reading `model_config.py` L238/L248 + `minicpm.py` L546-590 + `configs/minicpm.py`:

- `force_dense_minicpm=True` only overrides `has_sparse_attention` and `sparse_layer_ids`. It **does not** touch `mixer_types` / `lightning_layer_ids`.
- Layer construction in `MiniCPMDecoderLayer.__init__` dispatches purely on `self.mixer_type` (per-layer string from `config.mixer_types[layer_id]`):

| Per-layer mixer_type | Module | Backend | Effect of `--force-dense-minicpm` |
|---|---|---|---|
| `"minicpm4"` | `MiniCPMAttention` (softmax QKV) | The configured attention_backend | Backend rewrite: `minicpm_flashinfer → flashinfer` ⇒ **runs as dense softmax**, no top-k, no sparse_page_table |
| `"lightning"` / `"lightning_attn"` | `MiniCPMLightningMixer` | `SimpleGLAAttnBackend` (linear-attn / chunked GLA) | **Untouched** — lightning is linear attention with hidden state; has no equivalent dense softmax form. The chunked-GLA kernel still runs |
| `"attention"` / `"standard"` | `MiniCPMAttention` (softmax) | The configured attention_backend | Same as minicpm4: runs on stock FlashInfer dense |

Conclusion: the **user's intuition was half-right**. Sparse softmax layers do convert to dense; lightning layers do **NOT** — they keep running through `SimpleGLAAttnBackend`. Whether our specific submission model has any lightning layers needs `config.json` inspection (user says expectation is 8 sparse + 24 lightning; pending fcloud inspection).

### User input (champions context)
> "Almost all champion posts use W4A16 + mixed FP4/FP8 KV cache. They didn't mention force-dense or sparse — I guess they use sparse default (otherwise they would mention)."

This is a strong hint that the *winning* recipe might be **sparse + mixed-precision KV** rather than dense + uniform FP8 KV. Plausible, but with two caveats:
1. We have direct evidence that **GPTQ + FP8 KV + sparse breaks accuracy** (Test 8b: 76.07%, far below 99% normalized threshold). The descale-rounding compounding in top-k is real.
2. Champion posts may be using **per-layer mixed KV**: bf16 (or FP8 with very careful scaling) on the few sparse top-k-scoring layers, FP4/FP8 on the rest. We don't currently have this mixed-KV plumbing.

### Decision context
User wants to **retest GPTQ + FP8 KV + sparse**:
- (a) to confirm the Test 8b accuracy regression on the new fcloud / current code (the model has been re-quantized since, and the kernels updated);
- (b) to collect S1/S8/Smax for the sparse path, which we never did;
- (c) to validate or refute the champion-recipe hypothesis.

### Outcomes / pending
- This Q&A documented in chat log (this entry).
- User is starting fcloud so agent can inspect `config.json` of `MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` to confirm `mixer_types` (expected 8 minicpm4 + 24 lightning).
- Next: agent will propose a concrete retest plan (sparse + FP8 KV: accuracy + speed S1/S8/Smax) once mixer_types is confirmed.

### Cross-references
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) Tests 8b, 9, 12
- [model_config.py L238 / L248](../../../python/sglang/srt/configs/model_config.py#L238)
- [minicpm.py L546-L590](../../../python/sglang/srt/models/minicpm.py#L546-L590)
- [configs/minicpm.py L145-L167](../../../python/sglang/srt/configs/minicpm.py#L145-L167)
EOF

cat >> /home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md <<'EOF'

---

## Round 13d — 决策前讨论：稀疏的最优结果 + `--force-dense-minicpm` 的逐层行为

### 用户提问
1. `W4A16 + FP8 KV + sparse` 和 `W4A16 + bf16 KV + sparse` 我们历史最好的精度+速度结果是什么？
2. 加了 `--force-dense-minicpm` 之后，稀疏注意力层会以 dense softmax 运行（不再调 top-k 和稀疏算子）—— 已确认。那么 lightning 层呢？是否也变成 dense？

### Agent 回答

**Q1 —— 历史最好结果**

| 配置 | 最佳测试 | ori_acc | norm | C | mcq / qa / niah / cwe / fwe | 精度评测时长 | 备注 |
|---|---|---|---|---|---|---|---|
| GPTQ + FP8 KV + **稀疏** | Test 8b (`9d3ecd168`) | 76.07% | 95.08% | **0** | 60 / 66.67 / 96.67 / 60.33 / 96.67 | 2411 s | FP8 KV 破坏稀疏 top-k 评分（descale 舍入误差）|
| GPTQ + bf16 KV + **稀疏** | Test 9 (`79e49f39f`) | **79.67%** | **99.58%** | **1.0** | 63.33 / 56.67 / 100 / 81.67 / 96.67 | 3157 s | 唯一达到 C=1.0 的稀疏配置 |
| GPTQ + FP8 KV + **dense**（当前 baseline）| Test 12 (`9e82efe43`) | 79.29% | 99.11% | **1.0** | 63.33 / 72 / 97.78 / 100 / 63.33 | 4244 s | S1=121.71s, S8=44.09s, Smax=35.86s |

关键观察：**两个稀疏配置都没单独跑过 S1/S8/Smax**。表里 "时长" 是精度评测（150 样本 concurrency=8）的总时长，不是速度 benchmark trio。Test 8b 的 2411s < Test 12 的 4244s 不能直接推出 "稀疏更快"，差异多半来自输出长度（精度回退时模型答错或截断会更快结束，并不是 decode 真的快）。

用户提出 Test 8b 的时长短可能预示更高吞吐。反驳：dense FP8 已经 >100 tok/s，2411/4244 的比值由输出长度主导，不是 decode 速度。

**Q2 —— `--force-dense-minicpm` 下的逐层行为**

读 `model_config.py` L238/L248 + `minicpm.py` L546-590 + `configs/minicpm.py`：

- `force_dense_minicpm=True` 只改写 `has_sparse_attention` 和 `sparse_layer_ids`。**不会**动 `mixer_types` / `lightning_layer_ids`。
- `MiniCPMDecoderLayer.__init__` 的层构建严格按 `self.mixer_type`（来自 `config.mixer_types[layer_id]`）分支：

| 逐层 mixer_type | 模块 | 后端 | `--force-dense-minicpm` 的影响 |
|---|---|---|---|
| `"minicpm4"` | `MiniCPMAttention`（softmax QKV）| 配置的 attention_backend | 后端被改写：`minicpm_flashinfer → flashinfer` ⇒ **以 dense softmax 跑全 KV**，不调 top-k、不调 sparse_page_table |
| `"lightning"` / `"lightning_attn"` | `MiniCPMLightningMixer` | `SimpleGLAAttnBackend`（线性注意力 / chunked GLA）| **不动** —— lightning 是带隐状态的线性注意力，没有 dense softmax 等价形式。chunked-GLA kernel 照常执行 |
| `"attention"` / `"standard"` | `MiniCPMAttention`（softmax）| 配置的 attention_backend | 同 minicpm4：走 stock FlashInfer dense |

结论：**用户直觉对了一半**。稀疏 softmax 层会变成 dense；lightning 层 **不会** —— 仍走 `SimpleGLAAttnBackend`。我们提交模型究竟有没有 lightning 层得看 `config.json`（用户预期 8 个 sparse + 24 个 lightning，待 fcloud 起来后实测）。

### 用户输入（冠军帖子上下文）
> "几乎所有冠军帖都用 W4A16 + 混合 FP4/FP8 KV cache，他们没提 force-dense 也没说稀疏 —— 我猜他们默认用稀疏（否则会强调）。"

这是个强烈的暗示：**赢家配方很可能是稀疏 + 混合精度 KV**，不是 dense + 单一 FP8 KV。但要注意两点：
1. 我们有直接证据 **GPTQ + FP8 KV + 稀疏会破坏精度**（Test 8b 76.07%，远低于 99% 归一化阈值）。top-k 上的 descale 舍入误差累积是真的。
2. 冠军方案可能用 **逐层混合 KV**：在做 top-k 评分的几层稀疏层上用 bf16（或精心 scale 的 FP8），其它层用 FP4/FP8。我们目前没有这套混合 KV plumbing。

### 决策上下文
用户想 **重测 GPTQ + FP8 KV + 稀疏**：
- (a) 在新 fcloud / 当前代码上确认 Test 8b 的精度回退是否依然存在（模型已经重新量化过，kernel 也更新过）；
- (b) 收集稀疏路径的 S1/S8/Smax —— 我们从来没跑过；
- (c) 验证或证伪冠军方案猜想。

### 结论 / 待办
- 本轮 Q&A 已记录到 chat log（本条目）。
- 用户开 fcloud，等会让 agent 看 `MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` 的 `config.json`，确认 `mixer_types`（预期 8 个 minicpm4 + 24 个 lightning）。
- 下一步：mixer_types 确认后，agent 给出具体的重测方案（稀疏 + FP8 KV：精度 + 速度 S1/S8/Smax）。

### 交叉引用
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) Tests 8b、9、12
- [model_config.py L238 / L248](../../../python/sglang/srt/configs/model_config.py#L238)
- [minicpm.py L546-L590](../../../python/sglang/srt/models/minicpm.py#L546-L590)
- [configs/minicpm.py L145-L167](../../../python/sglang/srt/configs/minicpm.py#L145-L167)
EOF

cd /home/oldzhu/sglang && git add docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md && git commit -m "docs(chat R13d): pre-decision Q&A on sparse history + force_dense per-layer effect" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && git status --short && tail -20 docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md

Ran terminal command:  cd /home/oldzhu/sglang && git log --oneline -3 && git status docs/soar_2026_changes/chat/

Documented and pushed as commit `e4697078a`.

Now, regarding fcloud: ready when you are. Once you confirm fcloud is up I'll inspect `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/config.json` (the `mixer_types`, `sparse_layer_ids`, `lightning_layer_ids` fields) to confirm the layer split before we draft the retest plan.

While waiting, three things worth thinking about:

**1. Test 8b's "2411s" is most likely an *accuracy regression artifact*, not a speedup signal.**
- Sparse FP8 KV got mcq=60%, qa=66.67%, cwe=60.33% — far below dense FP8 baselines (mcq=63.33%, qa=72%, cwe=100%). When the model gives wrong/early-terminated answers, total decoded tokens drop ⇒ eval finishes faster.
- To prove "sparse is faster", we need S1/S8/Smax which decouples speed from correctness.

**2. The champion-recipe hypothesis ("W4A16 + mixed FP4/FP8 KV + sparse") deserves a careful read.**
- "Mixed" probably means **per-layer**: bf16 (or carefully calibrated FP8) on the few sparse top-k-scoring layers, FP4 elsewhere — so the descale-rounding compounding gets isolated.
- Or it could mean **K vs V different precision** (commonly K higher precision than V because K is consumed by the dot-product score).
- We currently support uniform `kv_cache_dtype` only. Mixed-KV plumbing would be CHANGE-level work, but maybe not too heavy if it lives at the pool level.

**3. The retest itself is cheap.**
- Reconfigure `prepare_env.sh`: drop `--force-dense-minicpm`, keep `--kv-cache-dtype fp8_e5m2`. Maybe restore `--attention-backend minicpm_flashinfer`.
- One accuracy run (~50min) + S1/S8/Smax (~5min total) = one fcloud session.

Let me know when fcloud is up and I'll inspect the config first.Continued with Autopilot: Task not yet complete


