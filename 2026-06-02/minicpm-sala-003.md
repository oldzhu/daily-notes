The ndt=2 refactor produced **catastrophic accuracy regression**: ori_accuracy = 15.13%, mcq = 0% (every MCQ generating 64460 tokens — runaway). The kernel reports accept_rate 0.73 (speculation is "working") but the model output is garbage. This is silent corruption worse than the ndt=1 case.

Let me pause the instance to save cost while we analyze. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -10

Paused. Now let me record results and analyze the regression.  

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#280-280), lines 280 to 460

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#460-460), lines 460 to 495

Searched for regex `positions.*seq_lens|prepare_for_verify|TARGET_VERIFY` (`**/python/sglang/srt/speculative/ngram_worker.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#130-130), lines 130 to 210

I now need to record this failure, propose a revert, and end the round. Let me update tracking docs. 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#1-1), lines 1 to 40

Searched for text `CHANGE_0164` (`**/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`), no results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md#40-40), lines 40 to 90

Replacing 1 lines with 2 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/TEST_RESULTS_TRACKING.md)


```

```
Now append a "Result" section to CHANGE_0164 docs and create a chat log.  

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md#1-1), lines 1 to 10

Ran terminal command:  cd /home/oldzhu/sglang && wc -l docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md#155-155), lines 155 to 167

Ran terminal command:  cd /home/oldzhu/sglang && cat >> docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md <<'EOF'

---

## Result (2026-05-13) — CATASTROPHIC, REVERT REQUIRED

**Test ID**: `Stage3b-ndt2-CATASTROPHIC` (commit `4b442f421`, fcloud `ai-e7e98a7c52`).

| Metric | Value | vs Baseline (Test 12 = 79.29%) |
|--------|-------|-------------------------------|
| ori_accuracy | **15.13 %** | −64.16 pt |
| normalized   | **18.92 %** | well below 97 % → **C = 0 (eliminated)** |
| mcq          | **0.00 %**  | runaway: avg_out = 64460 (full max_out_len) |
| niah         | 3.33 %      | −96.67 pt |
| cwe          | 15.67 %     | −56.33 pt |
| fwe          | 20.00 %     | −78.89 pt |
| qa           | 36.67 %     | −26.66 pt |
| eval duration| 7911 s (2 h 12 m) | ~2.6× baseline |

Kernel reported `accept_len = 1.46, accept_rate = 0.73` throughout the run —
i.e. the speculative pipeline was *structurally* alive, but the committed
tokens are wrong. Every MCQ sample exhausts the 65536 max_out_len budget,
strongly suggesting the bonus-position output token is corrupted (no stop
token ever emitted).

### Hypotheses (not yet bisected)

1. **Position-of-bonus mismatch.** We compute
   `positions = seq_lens + arange(ndt)`, expecting the bonus to occupy
   slot `seq_lens` and the draft to occupy `seq_lens+1`. NGRAM's own
   `_prepare_for_speculative_decoding` uses
   `reconstruct_indices_from_tree_mask(..., batch.seq_lens, positions, ...)`
   which derives positions from the tree mask. If the NGRAM convention
   expects the *root* of the verify tree to be at position `seq_lens-1`
   (overwriting the last-committed-token slot) rather than `seq_lens`,
   our bonus KV is written at the wrong row and every subsequent step
   reads stale state.
2. **Hidden distribution mismatch for the trained head.** The head was
   retrained on hidden states captured from the ndt=1 path
   (`hidden[:,0,:]` of a 1-token TARGET_VERIFY). In ndt=2 the position-0
   hidden has different attention context (it now sees one extra "future"
   token via tree-mask routing if the mask is wrong, or is identical if
   correct). If the captured-vs-served distributions diverge, the head
   produces nonsense drafts; but that alone would just hurt accept rate,
   not destroy the bonus output. So this can't be the *sole* cause.
3. **`prepare_for_verify` side-effects on `seq_lens`.** The shared
   `NgramVerifyInput.prepare_for_verify` may bump `seq_lens` by `ndt`
   internally; if our explicit `positions` was already computed before
   that call, the actual forward writes KV at offset `+ndt` past where
   we intended.

### Decision: REVERT

- Revert `medusa_worker.py` in both `python/sglang/srt/speculative/` and
  `benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/` to
  the Stage 3a `ndt=1` path (commit `3a15a6de3` /
  `Stage3a-force-dense` baseline: 78.40 % acc, S1=204.86 s).
- Keep CHANGE_0164 documents in place as a record of the failure.
- File follow-up investigation: instrument `_forward_verify_k1` to dump
  one micro-batch's `(positions, seq_lens_in, seq_lens_out, draft_tokens,
  input_ids_used_by_attention, committed_token)` quadruple and compare
  byte-for-byte against a single-token NgramWorker decode for the same
  prompt, to localize the off-by-one before re-attempting ndt=2.

### Rollback commands

```
git revert 4b442f421       # or
git checkout 3a15a6de3 -- python/sglang/srt/speculative/medusa_worker.py \
                          benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py
```
EOF
echo "appended en"

Ran terminal command:  cd /home/oldzhu/sglang && cat >> docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md <<'EOF'

---

## 结果（2026-05-13）—— 灾难性失败，必须回退

**测试 ID**：`Stage3b-ndt2-CATASTROPHIC`（commit `4b442f421`，fcloud `ai-e7e98a7c52`）。

| 指标 | 数值 | 与基线（Test 12 = 79.29%）对比 |
|------|------|------------------------------|
| ori_accuracy | **15.13 %** | −64.16 pt |
| normalized   | **18.92 %** | 远低于 97 % → **C = 0（淘汰）** |
| mcq          | **0.00 %**  | 失控：平均输出 64460 tokens（吃满 max_out_len）|
| niah         | 3.33 %      | −96.67 pt |
| cwe          | 15.67 %     | −56.33 pt |
| fwe          | 20.00 %     | −78.89 pt |
| qa           | 36.67 %     | −26.66 pt |
| 评测时长     | 7911 s（2 h 12 m）| 约为基线 2.6× |

kernel 一路报告 `accept_len = 1.46, accept_rate = 0.73`，结构上推测流水正常，但提交的 token 是错的。每条 MCQ 都把 65536 max_out_len 吃满，强烈暗示 bonus 位置输出的 token 已被破坏（永远输出不到 stop token）。

### 假设（尚未二分定位）

1. **bonus 的 position 不对**：我们用 `positions = seq_lens + arange(ndt)`，预期 bonus 在 `seq_lens` 槽位、draft 在 `seq_lens+1` 槽位。NGRAM 自带的 `_prepare_for_speculative_decoding` 调用 `reconstruct_indices_from_tree_mask(..., batch.seq_lens, positions, ...)` 从 tree mask 反推 positions。如果 NGRAM 约定 verify 树的 root 应位于 `seq_lens-1`（覆盖最后已提交 token 的槽位），那我们的 bonus KV 写到了错的行，后续读到的全是脏数据。
2. **trained head 的 hidden 分布不匹配**：head 是在 ndt=1 路径上采集的 `hidden[:,0,:]` 上重训的。在 ndt=2 下，position-0 hidden 的 attention 上下文不同；若 tree_mask 设置有偏差，捕获到的训练分布与在线推理分布发散，会让 head 草稿胡说。但单凭这点只会让 accept rate 变差，不应破坏 bonus 输出，所以这不是单独原因。
3. **`prepare_for_verify` 对 `seq_lens` 的副作用**：共享的 `NgramVerifyInput.prepare_for_verify` 内部可能把 `seq_lens` 加上 `ndt`。如果我们的 `positions` 是在调用之前就算好的，那真正的 forward 写 KV 时会比预期多偏移 `ndt`。

### 决策：回退

- 把 `medusa_worker.py`（两份拷贝）回到 Stage 3a `ndt=1` 路径（commit `3a15a6de3` / `Stage3a-force-dense` 基线：78.40 % acc，S1=204.86 s）。
- 保留 CHANGE_0164 文档作为失败记录。
- 立项后续调查：在 `_forward_verify_k1` 中 dump 一次 micro-batch 的 `(positions, seq_lens_in, seq_lens_out, draft_tokens, input_ids_used_by_attention, committed_token)` 五元组，与单 token NgramWorker decode 同 prompt 逐字节比对，先定位 off-by-one 再重新尝试 ndt=2。

### 回退命令

```
git revert 4b442f421       # 或
git checkout 3a15a6de3 -- python/sglang/srt/speculative/medusa_worker.py \
                          benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py
```
EOF
echo "appended zh"

Now create the chat log and commit.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_medusa-stage3b-ndt2-result_20260513_1000.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_medusa-stage3b-ndt2-result_20260513_1000.zh.md)

Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/TEST_RESULTS_TRACKING.md docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.en.md docs/soar_2026_changes/CHANGE_0164_medusa_stage3b_k1_ndt2_refactor.zh.md docs/soar_2026_changes/chat/CHAT_medusa-stage3b-ndt2-result_20260513_1000.en.md docs/soar_2026_changes/chat/CHAT_medusa-stage3b-ndt2-result_20260513_1000.zh.md && git status --short

Ran terminal command:  cd /home/oldzhu/sglang && git commit -m "CHANGE_0164: record catastrophic ndt=2 regression (acc 15.13%, C=0, MCQ runaway)" -m "Eval result on fcloud commit 4b442f421: ori_accuracy 15.13%, normalized 18.92%, mcq 0% (avg_out 64460 = max_out_len), duration 7911s. Kernel reported accept_rate 0.73 but committed tokens are wrong — bonus-position output corrupted. Documents Result section in CHANGE_0164 docs, adds TEST_RESULTS_TRACKING row Stage3b-ndt2-CATASTROPHIC, and chat log. Next: revert ndt=2 refactor to Stage3a-force-dense (commit 3a15a6de3) before further investigation." && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -10

Pushed. Now I'll stop here for user direction on the revert (revert is destructive enough to ask first).

## Summary

**CHANGE_0164 ndt=2 validation → CATASTROPHIC failure**

| Metric | Result | Baseline (Test 12) |
|---|---|---|
| ori_accuracy | **15.13%** | 79.29% |
| normalized | 18.92% → **C=0 (eliminated)** | 99.11% → C=1.0 |
| mcq | **0.00%** (avg_out 64460 = max_out_len, every sample runaway) | 63.33% |
| niah / cwe / fwe / qa | 3.33 / 15.67 / 20.0 / 36.67 % | 100 / 72 / 97.78 / 63.33 % |
| eval duration | 7911 s (2h12m) | ~4244 s |
| kernel accept_rate | 0.73 (structurally "working") | n/a |

Speculative pipeline is structurally alive (accept_rate 0.73) but committed tokens are garbage — bonus-position output corrupted, MCQ never emits stop token. Worse than the v23 ndt=1 silent corruption.

**Actions taken:**
- fcloud paused after eval.
- Result documented in CHANGE_0164 EN/ZH, TEST_RESULTS_TRACKING.md (row `Stage3b-ndt2-CATASTROPHIC`).
- Bilingual chat log written: CHAT_medusa-stage3b-ndt2-result_20260513_1000.en.md.
- Committed and pushed as `e3a194641` to `minicpm-src`.

**Recommended next step (awaiting your approval):** revert medusa_worker.py to the Stage 3a path from commit `3a15a6de3` (Stage3a-force-dense: 78.40% acc, S1=204.86s — best-known-good Medusa state). Three top suspects to investigate offline before retrying ndt=2: (1) bonus position off-by-one vs `NgramVerifyInput.prepare_for_verify`'s expected convention, (2) `seq_lens` mutation by `prepare_for_verify` invalidating our pre-computed `positions`, (3) tree_mask layout (we built it manually rather than via `reconstruct_indices_from_tree_mask`). Please confirm whether to proceed with the revert. 



