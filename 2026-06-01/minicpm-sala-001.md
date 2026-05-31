The key observation from the terminal: the crash at 03:10:31 (`available_size=15582175`) was from the **old server** (before CHANGE_0161 was applied on fcloud). The subsequent S1/S8/Smax tests all passed — that confirms CHANGE_0161 is working. Those 202.96/61.65/43.40s numbers are from the CHANGE_0161 run (no `--force-dense-minicpm`).

Now the new server with `SOAR_BACKEND_KEEP_FORCE_DENSE=1` timed out waiting — still booting. Let me wait longer. 

Starting: *Wait for server + run accuracy test* (2/5)

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1

Taking too long. Let me check the server logs to see what's happening. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py server-logs --lines 50 2>&1

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_bdrk_01RZ2LaLuKkMSa7QBJszqAd8__vscode-1778555277824/content.txt#1-1), lines 1 to 100

Read [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/chat-session-resources/c6c5f247-044e-42dd-8b77-ee67d33ffdf5/toolu_bdrk_01RZ2LaLuKkMSa7QBJszqAd8__vscode-1778555277824/content.txt#100-100), lines 100 to 200

Server is at `08:22:41` — still loading weights (torch.compile takes ~4-5 min). Let me wait one more time.  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py wait-server 2>&1

Server is up. Now run accuracy + all speed tests. 

Updated todo list

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py accuracy 2>&1

Accuracy test running. This takes ~25 minutes. I'll wait for the notification.

While waiting, here's the full **Stage 3b plan**:

---

## Stage 3b: Trained Medusa heads — complete plan

### Goal
Replace the zero-init draft in `MedusaWorker._forward_generate_k1` with a real trained `MedusaHead` network, achieving genuine `accept_rate > 0` and net speedup.

### Architecture (standard Medusa head)
```
MedusaHead(hidden_size=4096, vocab_size=122753):
    resblock: Linear(4096→4096, bias=False) + SiLU + residual
    lm_head:  Linear(4096→122753, bias=False)   # tied to base model LM head optionally
```
Weight size per head: `~4096×122753×2 + 4096×4096×2` ≈ **~1.03 GB BF16** for 1 head.

### Training pipeline

**Step 1 — Data collection** (on fcloud, ~10 min):
```bash
python3 scripts/medusa/collect_hidden_states.py \
    --model /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
    --data /root/data/perf_public_set.jsonl \
    --output /root/data/medusa_train_data.pt
```
Runs the base model forward on the eval data, saves `(hidden_states, labels)` pairs.

**Step 2 — Train heads** (~30-60 min on 1 GPU):
```bash
python3 scripts/medusa/train_medusa_heads.py \
    --base-model /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
    --train-data /root/data/medusa_train_data.pt \
    --num-heads 1 \
    --epochs 5 \
    --lr 1e-3 \
    --output /root/models/medusa_heads_k1.pt
```

**Step 3 — Load at server startup**:
- Add `--speculative-medusa-weights /root/models/medusa_heads_k1.pt` to `SGLANG_SERVER_ARGS`, or
- Pre-load via `preprocess_model.py` (save heads into the GPTQ checkpoint directory)

**Step 4 — Wire into MedusaWorker**:
Replace the zero-init draft line in `_forward_generate_k1` with actual head forward:
```python
# Current (Stage 3a):
draft_token_ids = torch.zeros(bs, 1, dtype=torch.int64, device=...)
# Stage 3b:
with torch.no_grad():
    draft_logits = self.medusa_head(hidden_states)  # (bs, vocab_size)
    draft_token_ids = draft_logits.argmax(dim=-1, keepdim=True)
```

### Expected results with trained head
| Metric | Stage 3a (zero-init) | Stage 3b (trained, K=1) |
|--------|---------------------|------------------------|
| accept_rate | 0.50 (root always, draft never) | ~0.80 (root + ~60% draft) |
| spec_accept_length | 1.00 | ~1.60 |
| Speedup vs no-spec | −20% (overhead) | +20-30% expected |
| S1 duration | 202.96s | ~90s (est.) |

The speedup ceiling for K=1 at 60% accept rate ≈ `1/(1 - 0.6 × 1/2)` ≈ 1.43× on decode-bottlenecked requests.

### Submission size check
- GPTQ model: ~4.5 GB → too big to ship → already handled by `prepare_model.sh` on-site quantization
- Medusa heads (K=1, BF16): ~1.03 GB → fits within 2 GB limit if shipped as pre-built weights
- OR: train heads during `prepare_model.sh` execution time (within 5h window)
