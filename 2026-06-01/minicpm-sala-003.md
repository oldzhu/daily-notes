Based on the Stage 3b results, the root cause of 0% draft acceptance is clear: **distribution mismatch** — the head was trained on non-GPTQ hidden states, but the GPTQ-quantized server produces different hidden states. Let me first read the training script and medusa worker, then implement a fix. 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#1-1), lines 1 to 100

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#100-100), lines 100 to 280

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#280-280), lines 280 to 400

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#400-400), lines 400 to 422

Now I have a clear picture. The plan to fix Stage 3b:

**Root cause**: Head trained on non-GPTQ hidden states. Fix: collect GPTQ model's hidden states from the running server, then retrain.

**Implementation**:
1. Add `SOAR_MEDUSA_DUMP_HIDDEN` mode to medusa_worker.py — saves TARGET_VERIFY hidden states to disk from the live GPTQ server
2. Add `--hidden-dump-path` to train_medusa_head.py — trains head from collected GPTQ data (no model loading needed) 

Created 6 todos

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#145-145), lines 145 to 170

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#270-270), lines 270 to 330

Compacted conversation

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#330-330), lines 330 to 375

Now I have everything I need. Let me implement both changes: 

Replacing 36 lines with 102 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```
Now add the `_flush_hidden_dump` method — I'll insert it just before `_forward_verify_k1`: 

Searched for text `def _forward_verify_k1` (`**/medusa_worker.py`), 2 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#254-254), lines 254 to 262

Replacing 4 lines with 29 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```
Now add `--hidden-dump-path` support to train_medusa_head.py: 

Replacing 5 lines with 83 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py)


```

```
Now add the `--hidden-dump-path` argument and dispatch call in `main()`: 

Replacing 11 lines with 23 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py)


```

```
Now sync the changes to the submission copy of medusa_worker.py: 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py#145-145), lines 145 to 180

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py#270-270), lines 270 to 320

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py#320-320), lines 320 to 375

Now apply the same changes to the submission copy: 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py#225-225), lines 225 to 245

Replacing 37 lines with 119 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```
I also need to add the `# ----- BaseSpecWorker-compatible duck-typed interface -----` separator back after the new init code in the submission copy: 

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py#160-160), lines 160 to 215

Good. Let me verify both files have `import os`: 

Searched for text `import os` (`**/speculative/medusa_worker.py`), 2 results

Good. Now let me do a quick sanity check on the modified train script:  

Searched for text `_train_from_gptq_dump` (`**/benchmark/soar/demo_sala/train_medusa_head.py`), 2 results

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#415-415), lines 415 to 432

Let me verify the train script imports `F` and `sys`:  

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#1-1), lines 1 to 40

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/train_medusa_head.py#40-40), lines 40 to 65

All imports are already in place (`F`, `sys`, `os`). Now let me commit and push:  

Starting: *Sync submission copy, commit, push* (3/6)

Ran terminal command:  cd /home/oldzhu/sglang && git diff --stat HEAD

Ran terminal command:  cd /home/oldzhu/sglang && git add python/sglang/srt/speculative/medusa_worker.py benchmark/soar/demo_sala/sglang/python/sglang/srt/speculative/medusa_worker.py benchmark/soar/demo_sala/train_medusa_head.py && git commit -m "CHANGE_0164: add SOAR_MEDUSA_DUMP_HIDDEN to collect GPTQ hidden states for Medusa head retraining

Root cause of Stage 3b 0% accept rate: head was trained on non-GPTQ hidden
states.  GPTQ model (40 layers W4A16 quantization) produces different hidden
states, so head predictions never match target model output.

Fix (Step 1 of 2):
- medusa_worker.py: add SOAR_MEDUSA_DUMP_HIDDEN=<path> env var.  When set,
  TARGET_VERIFY hidden states are accumulated in FP16 and flushed to <path>
  every 50 batches (or when SOAR_MEDUSA_DUMP_MAX_ROWS reached, default 20000).
  lm_head.weight is saved once to <path>.lm_head_weight.pt.
- train_medusa_head.py: add --hidden-dump-path flag.  When provided, skips
  model loading entirely: loads dumped (N,4096) FP16 hidden states, computes
  training labels y=argmax(F.linear(h, lm_head_weight)) from GPTQ head,
  trains W1 using existing cross-entropy loop, saves checkpoint.

Workflow on fcloud:
  1. Start server with SOAR_MEDUSA_DUMP_HIDDEN=/root/gptq_hidden_collect.pt
     (and NO SOAR_MEDUSA_HEAD_PATH to stay in stable Stage 3a)
  2. Run S1 speed test to collect ~5000+ GPTQ hidden state vectors
  3. train_medusa_head.py --hidden-dump-path /root/gptq_hidden_collect.pt
     --output /root/medusa_head_k1_gptq.pt --epochs 5 --lr 1e-4
  4. Restart server with SOAR_MEDUSA_HEAD_PATH=/root/medusa_head_k1_gptq.pt
  5. Run speed tests -- expect accept_len > 1.0 and S1 speedup"

Ran terminal command:  cd /home/oldzhu/sglang && git push minicpm-src mixed_minicpm_cudagraph

Pushed (`baf8f78de..c7da86d42`). Now I need to start fcloud, run data collection, train, and test. But per the rules I need explicit user approval first.

**The fcloud workflow for CHANGE_0164** requires these steps (need your approval):

1. **Start fcloud instance** (`start-instance`)
2. **Sync** (`sync`) — git pull on fcloud to get `c7da86d42`
3. **Restart server** in dump mode — `SOAR_MEDUSA_DUMP_HIDDEN=/root/gptq_hidden_collect.pt`, NO `SOAR_MEDUSA_HEAD_PATH` (Stage 3a + collecting)
4. **Run S1 speed test** — 48 requests × ~100-300 decode steps = ~5000-14000 hidden vectors saved to `/root/gptq_hidden_collect.pt`
5. **Train new head** on fcloud:
   ```bash
   python3 /root/sglang-minicpm/benchmark/soar/demo_sala/train_medusa_head.py \
     --hidden-dump-path /root/gptq_hidden_collect.pt \
     --output /root/medusa_head_k1_gptq.pt \
     --epochs 5 --lr 1e-4
   ```
6. **Restart server** with new head: `SOAR_MEDUSA_HEAD_PATH=/root/medusa_head_k1_gptq.pt`
7. **Run S1/S8/Smax speed tests** — check `accept_len` > 1.0 → real speedup
8. **Pause fcloud**

> **Do you approve starting the fcloud instance and running this pipeline?** 

Completed: *Sync submission copy, commit, push* (3/6)

Code changes committed and pushed (commit `c7da86d42`). Here's a summary of what was implemented:

**CHANGE_0164 — Step 1 of 2: GPTQ hidden-state collection for Medusa head retraining**

**medusa_worker.py** (both main + submission copy):
- New `SOAR_MEDUSA_DUMP_HIDDEN=<path>` env var — when set, forces `CaptureHiddenMode.LAST` even in Stage 3a mode (no trained head), accumulates `(N,4096)` FP16 hidden states from every TARGET_VERIFY forward
- Saves `lm_head.weight` once to `<path>.lm_head_weight.pt`  
- Flushes to disk every 50 batches via new `_flush_hidden_dump()` method
- Stops collecting after `SOAR_MEDUSA_DUMP_MAX_ROWS` (default 20000)

**train_medusa_head.py**:
- New `--hidden-dump-path` flag — skips model loading entirely, computes labels `y = argmax(F.linear(h, lm_head_weight))` directly from GPTQ head, reuses existing `train()` loop

Waiting for your approval to start the fcloud pipeline (start-instance → collect → train → test). 



