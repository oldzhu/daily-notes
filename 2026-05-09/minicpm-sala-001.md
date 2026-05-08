Starting: *Document Q&A on force_dense_minicpm semantics* (1/5)

Step 1 — append the Q&A documentation. I'll add it to the existing chat log (Round 13 already there) as a follow-up Q&A round. 

Ran terminal command:  cat >> /home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md <<'EOF'

---

## Round 13b — Q&A: semantics of `--force-dense-minicpm` and dense-vs-sparse choice

### User asks
> Before proceeding 0132 or other, just want to know more: if we stop the rewrite `minicpm_flashinfer → flashinfer` when `kv_cache_dtype == "fp4_e2m1"`, does `--force-dense-minicpm` mean we won't use sparse attention (no top-k scoring) but use dense attention in flashinfer? Our baseline (v18) is W4A16 + FP8 KV + dense, not + sparse, because dense gives better speed locally. Why is dense better here when theoretically (especially long context) sparse should be better? Is FlashInfer just more optimized?

### Agent answer (verified against source)

**`--force-dense-minicpm` has TWO independent effects:**

1. **Config-level** (`python/sglang/srt/configs/model_config.py` L238 / L248):
   ```python
   has_sparse_attention → False
   sparse_layer_ids     → []
   ```
   ⇒ Zero sparse layers in the model graph. No top-k scoring, no sparse-layer routing, no sparse Lightning path. Every layer is dense.
2. **Backend-level** (`python/sglang/srt/server_args.py` L1521-1525):
   ```python
   if self.force_dense_minicpm:
       if self.attention_backend == "minicpm_flashattn":   self.attention_backend = "fa3"
       elif self.attention_backend == "minicpm_flashinfer": self.attention_backend = "flashinfer"
   ```
   ⇒ Custom MiniCPM backend swapped out for stock FlashInfer / FA3.

**v18 baseline = W4A16 + FP8 KV + 100% dense + stock FlashInfer.** Confirmed.

**Why dense beats sparse on our local benchmark — three stacking factors:**

1. **Local speed dataset is short.** `speed_{s1,s8,smax}.jsonl` is hand-built by us. Sparse attention's win comes from `O(top_k · L)` vs `O(L²)`, but the crossover is roughly 8-16K tokens. At 1-4K tokens, top-k scoring + gather/scatter + scheduling overhead **costs more than it saves**.
2. **Stock FlashInfer is highly optimized on Blackwell SM120.** Paged KV, fused FP8 dequant, contiguous TensorCore tile shapes (CHANGE_0125 added the SM120 Marlin tile table). MiniCPM custom sparse backend is mostly Triton — solid for correctness, but typically 20-40% behind FlashInfer on raw decode throughput.
3. **GPTQ + sparse is broken on accuracy.** Tests 5/6: ~50% acc on old fcloud. Test 8b: 76% with default top_k. The sparse path interacts badly with GPTQ-quantized QKV — top-k scoring is sensitive to quant noise. Test 9 (GPTQ + bf16 KV + sparse) was the only working sparse-with-GPTQ result at 79.67% acc. With FP8 KV + sparse + GPTQ the accuracy collapses.

**Officially**, the organizers announced the speed dataset has more long-context samples than ours. Theoretically sparse should help there — but unless we fix #3, we cannot ship it. That is why dense + FP8 KV + GPTQ is the production config.

### Implication for CHANGE_0132

If we skip the **backend** rewrite when `kv_cache_dtype == "fp4_e2m1"`:

- `has_sparse_attention` is **still False** (config-level effect of `force_dense_minicpm` is independent of the backend rewrite). No sparse layers re-appear. We still get pure dense.
- `MiniCPMAttentionBackend` stays selected, every layer calls dense attention through it, and the CHANGE_0131 FP4 plumbing finally runs.
- **Open risk**: `MiniCPMAttentionBackend`'s dense decode path on SM120 may be slower than stock FlashInfer's. If the FP4 KV memory savings (44%) don't translate into more concurrent requests / higher throughput than the kernel gap, the change is net-negative on speed even if it boots and is accurate.

### Decision

User chose the conservative path:

1. **First**, do a code review of `MiniCPMAttentionBackend.forward_decode` to assess whether its dense decode path is competitive with stock FlashInfer on SM120 — establish up-front whether we expect a speed cliff.
2. **Then**, implement Option A and measure on fcloud (smoke → accuracy → S1/S8/Smax).

### Cross-references

- [server_args.py L1521-1525](../../../python/sglang/srt/server_args.py#L1521-L1525) — backend rewrite
- [model_config.py L238](../../../python/sglang/srt/configs/model_config.py#L238) / [L248](../../../python/sglang/srt/configs/model_config.py#L248) — sparse layer disable
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) Tests 5/6/8b/9/12 — GPTQ sparse-vs-dense empirical history
- [CHANGE_0132_nvfp4_kv_force_dense_compat.en.md](../CHANGE_0132_nvfp4_kv_force_dense_compat.en.md) §3 Option A
EOF


Ran terminal command:  cat >> /home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md <<'EOF'

---

## Round 13b — Q&A：`--force-dense-minicpm` 语义与 dense vs sparse 选择

### 用户提问
> 在动 0132 之前再问一下：如果 `kv_cache_dtype == "fp4_e2m1"` 时不再把 `minicpm_flashinfer` 改写成 `flashinfer`，那 `--force-dense-minicpm` 是不是意味着我们不用 sparse attention（没有 top-k 打分）而走 flashinfer 的 dense？我们 v18 baseline 是 W4A16 + FP8 KV + dense（不是 + sparse），因为本地 dense 比 sparse 速度更好。但理论上长输入 sparse 应该更快，为什么这里 dense 更好？是 FlashInfer 优化得更狠吗？

### Agent 回答（已对源码核对）

**`--force-dense-minicpm` 有两个独立效果：**

1. **Config 层**（`python/sglang/srt/configs/model_config.py` L238 / L248）：
   ```python
   has_sparse_attention → False
   sparse_layer_ids     → []
   ```
   ⇒ 模型图里**完全没有 sparse 层**。没有 top-k 打分、没有 sparse 层路由、没有 sparse Lightning。所有层都是 dense。
2. **Backend 层**（`python/sglang/srt/server_args.py` L1521-1525）：
   ```python
   if self.force_dense_minicpm:
       if self.attention_backend == "minicpm_flashattn":   self.attention_backend = "fa3"
       elif self.attention_backend == "minicpm_flashinfer": self.attention_backend = "flashinfer"
   ```
   ⇒ MiniCPM 自定义后端被换成主线 FlashInfer / FA3。

**v18 baseline = W4A16 + FP8 KV + 100% dense + 主线 FlashInfer**。确认。

**为什么本地 dense 比 sparse 好 —— 三个叠加因素：**

1. **本地速度数据集太短**。`speed_{s1,s8,smax}.jsonl` 是我们手搓的。sparse 的胜利来自 `O(top_k · L)` vs `O(L²)`，但临界点大约在 8-16K token。在 1-4K token 上，top-k 打分 + gather/scatter + 调度开销 **比节省的还多**。
2. **主线 FlashInfer 在 Blackwell SM120 上优化得很好**。paged KV、fused FP8 dequant、连续的 TensorCore tile（CHANGE_0125 已加 SM120 Marlin tile 表）。MiniCPM sparse backend 主要是 Triton —— 正确性可靠，但 raw decode 吞吐通常比 FlashInfer 慢 20-40%。
3. **GPTQ + sparse 在精度上是坏的**。Test 5/6：旧 fcloud 上 ~50% 精度。Test 8b：默认 top_k 下 76%。sparse 路径和 GPTQ 量化的 QKV 配合差 —— top-k 打分对量化噪声敏感。Test 9（GPTQ + bf16 KV + sparse）是唯一能用的 sparse-with-GPTQ 结果，79.67%。FP8 KV + sparse + GPTQ 精度直接崩。

**官方** 已经宣布速度数据集比我们手搓的有更多长上下文样本。理论上 sparse 在那应该有用 —— 但只要 #3 不修，就不能上。所以生产配置必然是 dense + FP8 KV + GPTQ。

### 对 CHANGE_0132 的影响

`kv_cache_dtype == "fp4_e2m1"` 时跳过 **backend** 重写：

- `has_sparse_attention` **还是 False**（`force_dense_minicpm` 的 config 层效果与 backend 重写互相独立）。sparse 层不会回来。还是纯 dense。
- `MiniCPMAttentionBackend` 保持选中，每层都通过它走 dense attention，CHANGE_0131 的 FP4 plumbing 终于能跑起来。
- **未解风险**：`MiniCPMAttentionBackend` 的 dense decode 路径在 SM120 上可能比主线 FlashInfer 慢。如果 FP4 KV 节省的 44% 内存带来的并发提升 / 吞吐提升 抵不过 kernel 差距，那即使能启动且精度过，速度也是负收益。

### 决定

用户选保守路线：

1. **先**做 `MiniCPMAttentionBackend.forward_decode` 代码审查，事先评估它的 dense decode 在 SM120 上是否能跟主线 FlashInfer 抗衡 —— 看会不会有速度悬崖。
2. **然后**实现方案 A，在 fcloud 上量化（smoke → 精度 → S1/S8/Smax）。

### 交叉引用

- [server_args.py L1521-1525](../../../python/sglang/srt/server_args.py#L1521-L1525) —— backend 重写
- [model_config.py L238](../../../python/sglang/srt/configs/model_config.py#L238) / [L248](../../../python/sglang/srt/configs/model_config.py#L248) —— sparse 层关闭
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) Test 5/6/8b/9/12 —— GPTQ sparse-vs-dense 实测史
- [CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md](../CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md) §3 方案 A
EOF


Documentation done. Mark todo 1 complete and proceed to step 2 — code review of `MiniCPMAttentionBackend.forward_decode`.  

Starting: *Code review MiniCPMAttentionBackend.forward_decode* (2/5)

Searched for regex `def forward_decode|def forward_extend|flashinfer|sparse|dense|wrapper|begin_forward|run\(` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Searched for regex `def forward_decode|def forward_extend|def init_forward_metadata|is_sparse_layer|is_dense|BatchDecodeWithPagedKV|self\.decode_wrappers|run\(|begin_forward` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1130-1130), lines 1130 to 1280

Searched for regex `sparse_layer_ids|has_sparse_attention|is_sparse_layer|RadixAttention` (`**/python/sglang/srt/models/minicpm**`), 8 results

Searched for files matching `**/python/sglang/srt/models/minicpm_sala*.py`, no matches

Searched for regex `MiniCPMSALA|minicpm_sala` (`**/python/sglang/srt/models/**`), 4 results

Searched for regex `sparse_layer_ids|is_sparse|has_sparse|self\.attn\s*=|class .*Attention|def __init__.*layer_id` (`**/python/sglang/srt/models/minicpm.py`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#170-170), lines 170 to 270

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1280-1280), lines 1280 to 1490

Important finding. Let me confirm `forward_extend` to see whether there's any dense branch there.  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#911-911), lines 911 to 1130

Searched for regex `has_sparse_attention|sparse_layer_ids|num_sparse_topk_tokens|self\.sparse_topk\s*=|dense_len` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Major finding. Let me check the init path one more time to be 100% sure:   

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#219-219), lines 219 to 250

This is decisive. Step 2 (code review) just **invalidated Option A**. Let me update CHANGE_0132 to reflect the finding accurately, then surface it to the user before implementing anything that won't work.

Updating CHANGE_0132 EN:  

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0132_nvfp4_kv_force_dense_compat.en.md#1-1), lines 1 to 130

Replacing 50 lines with 143 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0132_nvfp4_kv_force_dense_compat.en.md)


```

```
Now mirror to ZH:   

Replacing 47 lines with 127 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md)


```

```
Now append a brief Round 13c entry to the chat log capturing the code-review verdict:    

Ran terminal command:  cat >> /home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md <<'EOF'

---

## Round 13c — code review verdict: Option A is INFEASIBLE

### User asks
> Document the answer for tracking. Do (1) [code review of `MiniCPMAttentionBackend.forward_decode`] first, then do (2) [run Option A and measure].

### Agent action
Read `python/sglang/srt/layers/attention/minicpm_backend.py` end-to-end. Four findings invalidate Option A as proposed:

1. **`MiniCPMSparseBackend.__init__` L226-230** raises `ValueError` if `has_sparse_attention=False`.
2. **`force_dense_minicpm=True` overrides `has_sparse_attention → False`** at config level (`model_config.py` L238).
3. **`forward_decode` L1130-1300** unconditionally runs `get_topk_for_sparse` + `sparse_kernel_extension.get_block_table_v3`. No dense branch.
4. **`forward_extend` L989+** routes both branches through `metadata.sparse_page_table`. Even short sequences traverse sparse plumbing.

⇒ The custom backend cannot even instantiate under `--force-dense-minicpm`, and even if it could, every code path is sparse top-k. My prior claim that "MiniCPMAttentionBackend already supports dense-only batches" was wrong.

### Outcomes
- **Option A (skip rewrite for FP4) is dead.** No work performed on it.
- **Option A′ (revised)** would require fork-level changes to `MiniCPMSparseBackend`: relax the init guard, add a `is_dense_run` branch in `forward_decode`/`forward_extend` that bypasses top-k and uses the full `page_table` directly through the already-imported `BatchDecodeWithPagedKVCacheWrapper`, and re-allocate cudagraph buffers accordingly. Effort: hundreds of LoC + careful cudagraph re-validation. Not a 1-day patch.
- **Option B** (FP4 KV in stock FlashInfer) requires upstream kernel templates — out of scope for the competition timeline.
- **Recommendation: Option C — park CHANGE_0131/0132.** Keep the four R13 hardening commits (`d4608f170`, `fd7e797ea`, `252cc4d64`, `8a0976593`) — they are inert at runtime when `SOAR_FP4_KV_CACHE=0` (default). Pivot to other catalog items.
- Updated `CHANGE_0132_nvfp4_kv_force_dense_compat.{en,zh}.md` with a new §8 "Round 13b code review" containing the L221-L230 / L1130-L1300 / L989-L1056 evidence and revised §7 to recommend Option C.

### Agent did NOT proceed to step (2)

Step 2 (run Option A and measure) is blocked because Option A is provably wrong. Surfaced finding to user before any fcloud spend.

### Cross-references
- [CHANGE_0132_nvfp4_kv_force_dense_compat.en.md](../CHANGE_0132_nvfp4_kv_force_dense_compat.en.md) §8
- [minicpm_backend.py L221-L230](../../../python/sglang/srt/layers/attention/minicpm_backend.py#L221-L230)
- [minicpm_backend.py L1130-L1300](../../../python/sglang/srt/layers/attention/minicpm_backend.py#L1130-L1300)
- [model_config.py L238](../../../python/sglang/srt/configs/model_config.py#L238)
EOF


Ran terminal command:  cat >> /home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md <<'EOF'

---

## Round 13c — 代码审查结论：方案 A 不可行

### 用户提问
> 把上面那个回答记录下来供以后参考。先做 (1)（审 `MiniCPMAttentionBackend.forward_decode`），然后再做 (2)（跑方案 A 测量）。

### Agent 操作
通读 `python/sglang/srt/layers/attention/minicpm_backend.py`，得到四个否定方案 A 的发现：

1. **`MiniCPMSparseBackend.__init__` L226-230**：`has_sparse_attention=False` 时直接 `raise ValueError`。
2. **`force_dense_minicpm=True` 在 config 层把 `has_sparse_attention` 强制为 False**（`model_config.py` L238）。
3. **`forward_decode` L1130-1300** 无条件调 `get_topk_for_sparse` + `sparse_kernel_extension.get_block_table_v3`，没有 dense 分支。
4. **`forward_extend` L989+** 两条分支都走 `metadata.sparse_page_table`。短序列也跳不开 sparse 管道。

⇒ 在 `--force-dense-minicpm` 下，自定义后端 **连初始化都失败**；即便能初始化，每条路径都是 sparse top-k。我之前说"MiniCPMAttentionBackend 已支持 dense-only batch"是错的。

### 结论
- **方案 A（FP4 时跳过重写）已死**。没做任何实现。
- **方案 A′（修订版）** 需要在 `MiniCPMSparseBackend` 里 fork：放宽 init 守卫，在 `forward_decode`/`forward_extend` 加 `is_dense_run` 分支跳过 top-k、直接用完整 `page_table` 走已经导入的 `BatchDecodeWithPagedKVCacheWrapper`，cudagraph buffer 重新分配。工作量：几百行 LoC + 仔细的 cudagraph 重验证。不是 1 天能搞定的。
- **方案 B**（主线 FlashInfer 加 FP4 KV）需要上游 kernel template —— 超出比赛时间线。
- **推荐：方案 C —— 冻结 CHANGE_0131/0132**。保留 R13 的四个加固 commit（`d4608f170`、`fd7e797ea`、`252cc4d64`、`8a0976593`），`SOAR_FP4_KV_CACHE=0`（默认）时它们在运行期是闲置的。转其他目录项。
- 已更新 `CHANGE_0132_nvfp4_kv_force_dense_compat.{en,zh}.md`：新增 §8 "Round 13b 代码审查"，列 L221-L230 / L1130-L1300 / L989-L1056 证据；§7 改为推荐方案 C。

### Agent 没有进入第 (2) 步

第 (2) 步（跑方案 A 测量）已无意义，因为方案 A 已被证伪。在花 fcloud 时间之前先把这个结论交给用户。

### 交叉引用
- [CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md](../CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md) §8
- [minicpm_backend.py L221-L230](../../../python/sglang/srt/layers/attention/minicpm_backend.py#L221-L230)
- [minicpm_backend.py L1130-L1300](../../../python/sglang/srt/layers/attention/minicpm_backend.py#L1130-L1300)
- [model_config.py L238](../../../python/sglang/srt/configs/model_config.py#L238)
EOF


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0132_nvfp4_kv_force_dense_compat.en.md docs/soar_2026_changes/CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md && git commit -m "docs(change_0132,chat): R13b/R13c code review - Option A infeasible

R13b chat log: documented the dual semantics of --force-dense-minicpm
(config-level: zero sparse layers; backend-level: stock FlashInfer)
and why dense beats sparse on the local benchmark (short dataset,
FlashInfer SM120 optimization, GPTQ+sparse accuracy break).

R13c code review: read minicpm_backend.py end-to-end and found four
issues that invalidate CHANGE_0132 Option A as proposed:
  1. MiniCPMSparseBackend.__init__ L226-230 raises ValueError when
     has_sparse_attention=False.
  2. force_dense_minicpm=True clamps has_sparse_attention to False
     at config level (model_config.py L238).
  3. forward_decode L1130-1300 unconditionally calls
     get_topk_for_sparse and sparse_kernel_extension.
  4. forward_extend L989+ routes both branches through
     sparse_page_table.

Conclusion: the custom backend cannot even instantiate under
--force-dense-minicpm, and has no dense codepath. My prior advice
was wrong.

CHANGE_0132 EN+ZH revised:
- Status: Option A INFEASIBLE.
- New Option A' (heavy fork) and Option C (park, recommended).
- New \u00a78 'Round 13b code review' with evidence.

No fcloud work performed. Surfacing finding to user before any
further spend." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Completed: *Code review MiniCPMAttentionBackend.forward_decode* (2/5)

## Summary

**Step 1 (documentation) — done.** Round 13b Q&A captured in chat log EN+ZH (semantics of `--force-dense-minicpm`, dense-vs-sparse history, three-factor explanation).

**Step 2 (code review) — done, with a major correction.** Reading minicpm_backend.py end-to-end revealed **CHANGE_0132 Option A as I proposed it is infeasible**:

| # | Finding | Evidence |
|---|---|---|
| 1 | Custom backend hard-rejects `has_sparse_attention=False` | `MiniCPMSparseBackend.__init__` L226-230: `raise ValueError(...)` |
| 2 | `force_dense_minicpm=True` clamps `has_sparse_attention → False` at config level | model_config.py L238 |
| 3 | `forward_decode` has **no dense branch** | L1130-1300: unconditional `get_topk_for_sparse` + `sparse_kernel_extension` |
| 4 | `forward_extend` both branches use `sparse_page_table` | L989-1056 |

⇒ Under `--force-dense-minicpm`, the custom backend cannot even instantiate. Even if we relaxed that, every code path runs sparse top-k. My earlier claim that "MiniCPMAttentionBackend already supports dense-only batches" was wrong, and I corrected it explicitly in the docs.

**Step 3 (Option A implementation + fcloud test) — NOT executed**, because Option A is provably wrong. Pushing through would have wasted fcloud spend on a dead config.

**Real options remaining:**
- **A′** — fork `MiniCPMSparseBackend` to add a real dense branch (relax init guard + `is_dense_run` flag in `forward_decode`/`forward_extend` + cudagraph buffer rework). Several hundred LoC + careful re-validation. Not 1 day.
- **B** — FP4 KV in stock FlashInfer. Out of scope (upstream kernel templates).
- **C** — **park CHANGE_0131/0132**, keep the four R13 hardening commits inert. Pivot to other catalog items. **Recommended.**

Committed as 64e180177 — CHANGE_0132 EN+ZH now has §8 "Round 13b code review" with line-citation evidence, status changed to "Option A INFEASIBLE", §7 recommends Option C. Chat log appended Round 13b + 13c.

**Decision needed from you:**
1. **Park (C)** and pivot — what next from the catalog (Marlin tile tuning, scheduling, speculative decoding)?
2. **Pursue A′** despite the cost? (multi-day kernel + cudagraph work, no certainty FP4 KV beats FP8 on speed/accuracy)
3. **Something else**?

fcloud is still down (no spend during this session beyond the doc-side work).  



