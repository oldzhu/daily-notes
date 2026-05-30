Let me check how to pass env overrides to the fcloud server restart, then start the instance. 

Searched for text `restart-server|restart_server|SGLANG_SERVER_ARGS|prepare_env` (`**/scripts/fcloud/fcloud_workflow.py`), 17 results

`restart-server` already supports `--env KEY=VAL`. Now let me start the instance and run the test. 

Created 5 todos

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py restart-server --env SOAR_BACKEND_KEEP_FORCE_DENSE=1 2>&1

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1

Server is still booting (CUDA graph capture + torch.compile takes ~4 min). While it warms up, here's the **next-steps plan** after this test:

---

## Post-test plan: Medusa Stage 3b roadmap

### **Immediate (today): Decision point from this test**

| Result | Action |
|--------|--------|
| `--force-dense-minicpm` restores accuracy to ~79% AND no crashes | Change default to `SOAR_BACKEND_KEEP_FORCE_DENSE=1` permanently (structural fix better than CHANGE_0158/0159/0161 patches) |
| Speed same or better than current (flashinfer without force-dense) | Double win — cleaner code path AND stable |
| Speed slightly worse | Keep current patches, keep `SOAR_BACKEND_KEEP_FORCE_DENSE=0`, still structurally safe after CHANGE_0161 |

---

### **Stage 3b: Trained Medusa heads — the real speedup**

The current K=1 zero-init heads always reject (accept_rate=0). To get actual speedup we need trained heads with real acceptance rate > 0.

**Two options for getting trained weights:**

#### Option A — Use publicly available pre-trained Medusa heads
MiniCPM-4 is a public model. Check if the community has released Medusa heads for it:
- Search HuggingFace for `MiniCPM-4 medusa` or `MiniCPM-SALA medusa`
- Likely doesn't exist yet (model is competition-specific)

#### Option B — Train Medusa heads ourselves (most likely path)

The standard Medusa training pipeline:
1. **Data**: Use `perf_public_set.jsonl` (same tasks as eval) + any general text corpus
2. **Architecture**: 1 MedusaHead per speculative head = linear layer `(hidden_size → vocab_size)` with ResNet block
3. **Training**: Freeze the base model weights, train only the head(s) on next-token prediction with teacher forcing
4. **Loss**: Cross-entropy on head_k predicting token at position `t+k+1`
5. **Duration**: ~1-2 hours on a single GPU for a small head

**Key parameters to tune:**
- `speculative_num_medusa_heads` (K): more heads = more draft tokens per step, higher ceiling but lower accept rate per head
- `speculative_num_draft_tokens` (must = K+1): controls speculation depth
- Head architecture depth: 1 ResNet block is standard

**Expected accept rates with trained heads:**
- K=1: ~60-70% (1 draft token accepted ~65% of the time → `spec_accept_length ≈ 1.65` → ~1.65x speedup ceiling)
- K=2: ~50-60% on head-1, ~40% on head-2 → geometric: `1 + 0.55 + 0.55*0.45 ≈ 1.8` → higher ceiling
- Real-world with tree-based verification: 1.5-2x speedup expected

**What we need to implement for Stage 3b:**
1. Load `medusa_head.pt` weights at server startup (add to `prepare_model.sh`)
2. Implement actual `MedusaHead.forward()` in `MedusaWorker._forward_generate_k1` (replace zero-init draft with real head inference)
3. Verify tree-expansion / multi-head path works with our GLA state management
4. Run training script (can use sglang's built-in `python -m sglang.train.medusa` or write a lightweight one)

**Submission size consideration**: trained Medusa heads for 1 head ≈ `hidden_size × vocab_size` ≈ `4096 × 122753 × 2 bytes (BF16)` ≈ **~1 GB** — fits within the 2GB submission limit if we keep K=1 or K=2.

---

Now let me wait for the server. 

Checked terminal output
