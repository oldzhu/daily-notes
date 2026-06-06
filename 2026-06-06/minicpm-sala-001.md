Compacted conversation

Starting: *Iter3: add pre_prepare_for_verify dump phase* (9/13)

Searched for regex `preflight_dump|prepare_for_verify` (`**/python/sglang/srt/speculative/{medusa_worker,ngram_worker}.py`), 12 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#180-180), lines 180 to 280

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#280-280), lines 280 to 360

Now I have what I need. Add `pre_prepare_for_verify` dumps just before each `prepare_for_verify` call in both workers. 

Replacing 12 lines with 24 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py)


```

```


Replacing 5 lines with 17 lines in [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py)


```

```
Commit and push, then start fcloud and run preflight. 

Ran terminal command:  cd /home/oldzhu/sglang && git add -u && git commit -m "preflight iter3: add pre_prepare_for_verify dump in both workers" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Pushed `3d2435d26`. Now resume fcloud and run the preflight loop. 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py start-instance 2>&1 | tail -10

Ran terminal command:  cd /home/oldzhu/sglang && for i in 1 2 3 4 5 6 7 8 9 10; do python3 scripts/fcloud/fcloud_exec.py exec 'echo ready' 2>&1 | tail -3 | grep -q ready && echo "READY after ${i}0s" && break; sleep 10; done

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py sync 2>&1 | tail -15

Synced cleanly. Now run preflight ngram + medusa:  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 900 'cd /root/submission_sim && bash ./preflight_drive.sh ngram 2>&1 | tail -15' 2>&1 | tail -20

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 900 'cd /root/submission_sim && bash ./preflight_drive.sh medusa 2>&1 | tail -15' 2>&1 | tail -20

Medusa probe succeeded (200 OK) before the post-idle KV leak. Now diff: 

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 'ls -la /tmp/dump_*.pkl' 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 120 'cd /root/sglang-minicpm && python3 benchmark/soar/demo_sala/preflight_diff.py --ngram /tmp/dump_ngram.pkl --medusa /tmp/dump_medusa.pkl 2>&1 | tee /tmp/iter3_diff.txt | tail -200' 2>&1 | tail -210

The diff didn't show `pre_prepare_for_verify` phase — let me check whether the diff tool printed it:  

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 'grep -E "PHASE|pre_prepare" /tmp/iter3_diff.txt' 2>&1 | tail -20

Searched for regex `phase|PHASE` (`**/benchmark/soar/demo_sala/preflight_diff.py`), 19 results

Replacing 1 lines with 1 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preflight_diff.py)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add -u && git commit -m "preflight_diff: include pre_prepare_for_verify phase" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 120 'cd /root/sglang-minicpm && git pull --ff-only origin mixed_minicpm_cudagraph 2>&1 | tail -3 && python3 benchmark/soar/demo_sala/preflight_diff.py --ngram /tmp/dump_ngram.pkl --medusa /tmp/dump_medusa.pkl --phase pre_prepare_for_verify 2>&1 | tee /tmp/iter3_pre_prep.txt' 2>&1 | tail -80

Searched for text `choices=` (`**/benchmark/soar/demo_sala/preflight_diff.py`), 1 result

Replacing 1 lines with 1 lines in [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preflight_diff.py)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add -u && git commit -m "preflight_diff: add pre_prepare_for_verify to argparse choices" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 120 'cd /root/sglang-minicpm && git pull --ff-only origin mixed_minicpm_cudagraph 2>&1 | tail -3 && python3 benchmark/soar/demo_sala/preflight_diff.py --ngram /tmp/dump_ngram.pkl --medusa /tmp/dump_medusa.pkl --phase pre_prepare_for_verify 2>&1 | tee /tmp/iter3_pre_prep.txt' 2>&1 | tail -80

**Critical finding** — H1/H3 confirmed, H2 ruled out. At `pre_prepare_for_verify` (BEFORE `prepare_for_verify` runs):

- **ngram**: `seq_lens=[7]`, `out_cache_loc=[1,2,3,4,5,6,7]` — fresh EXTEND with 7 prompt slots, no bonus
- **medusa**: `seq_lens=[8]`, `out_cache_loc=[8]` — already +1 with a single bonus slot committed at index 8

So `prepare_for_verify` itself is innocent. The bonus-token KV write happens **upstream** in MedusaWorker's main-decode/extend flow. Pause fcloud, document, then propose iter 4 (locate & fix the bonus commit).   

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 60 'base64 /tmp/iter3_pre_prep.txt' 2>&1 | tail -50 > /tmp/iter3_pre_prep.b64 && wc -l /tmp/iter3_pre_prep.b64 && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_workflow.py pause-instance 2>&1 | tail -5

Paused. Now let me trace where the bonus KV slot is committed upstream — this will inform iter 4's fix proposal. Need to find what happens between EXTEND completion and the verify-prep entry in MedusaWorker.   

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/medusa_worker.py#100-100), lines 100 to 270

Searched for regex `is_extend|bonus|seq_lens \+|out_cache_loc` (`**/python/sglang/srt/speculative/eagle_worker.py`), 40 results

Searched for regex `is_extend|prepare_for_extend|spec_algorithm` (`**/python/sglang/srt/speculative/ngram_worker.py`), 6 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/speculative/ngram_worker.py#130-130), lines 130 to 170

Looking at workers:
- **NgramWorker** EXTEND: returns early but **keeps** `spec_algorithm=NGRAM`
- **MedusaWorker** EXTEND: explicitly sets `batch.spec_algorithm = SpeculativeAlgorithm.NONE`

That's almost certainly the bug — the scheduler's post-extend bookkeeping treats `NONE` like a regular extend (advances seq_lens + commits one slot for the just-decoded token) but treats `NGRAM` as spec (defers KV commit to `prepare_for_verify`). Let me verify: 

Searched for regex `spec_algorithm.is_(none|ngram)|prepare_for_decode|seq_lens \+= 1|seq_lens\.add_` (`**/python/sglang/srt/managers/scheduler*.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py#365-365), lines 365 to 410

Searched for regex `def prepare_for_decode|spec_algorithm.is_none|seq_lens.*\+= 1|seq_lens.add_` (`**/python/sglang/srt/managers/schedule_batch.py`), 12 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/managers/schedule_batch.py#1939-1939), lines 1939 to 2020

**Iter 4 fix candidate isolated.** Found the exact mechanism in schedule_batch.py:

```python
def prepare_for_decode(self):
    ...
    if not self.spec_algorithm.is_none():
        return  # spec workers handle decode-prep themselves
    # else: allocate 1 slot, advance seq_lens by 1, bump kv_committed_len
```

So: NgramWorker's extend leaves `spec_algorithm = NGRAM` → next iteration's `prepare_for_decode` takes the early return → seq_lens stays at 7. MedusaWorker's extend does `batch.spec_algorithm = SpeculativeAlgorithm.NONE` → next iteration's `prepare_for_decode` falls through → allocates 1 slot, advances seq_lens to 8.

**One-line fix** in medusa_worker.py: drop the `spec_algorithm = NONE` line in the EXTEND branch (NgramWorker doesn't do this and is the canonical reference).

Now write §5.3 docs documenting iter 3 findings + iter 4 proposal: 

Searched for text `## 6. Risks` (`**/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md`), 1 result

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md#195-195), lines 195 to 208

Replacing 5 lines with 80 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.en.md)


```

```
Now the ZH doc. First find the anchor:  

Searched for regex `## 6\.|89a30d5d0` (`**/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md`), 3 results

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md#200-200), lines 200 to 209

Replacing 7 lines with 80 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.zh.md)


```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0165_medusa_preflight_diff.{en,zh}.md && git commit -m "preflight: CHANGE_0165 §5.3 iter3 result + §5.4 iter4 plan (root cause = spec_algorithm reset in MedusaWorker EXTEND)" && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -3

Starting: *Iter4: drop spec_algorithm reset in MedusaWorker EXTEND (awaiting user go)* (4/5)

## Iter 3 result — root cause precisely located

**Diff progression**: iter 1: 13 fields → iter 2: 5 fields → iter 3: same 5 pre_verify fields, **plus 3 pre_prepare_for_verify fields** that prove the bug is upstream of `prepare_for_verify`.

**The mechanism** (one-line bug, traced through three files):

| layer | code |
|---|---|
| medusa_worker.py EXTEND branch | `batch.spec_algorithm = SpeculativeAlgorithm.NONE` ← **this line** |
| schedule_batch.py `prepare_for_decode` | `if not self.spec_algorithm.is_none(): return` |
| ngram_worker.py EXTEND branch | (no reset — leaves NGRAM) |

So next iteration's `prepare_for_decode` falls through for Medusa only → allocates 1 slot, `seq_lens += 1`, `kv_committed_len += 1`. By the time MedusaWorker re-enters DECODE, the bonus slot is already committed. Same pre-extend state, +1 KV slot before verify-prep.

**Iter 4 proposal (one-line fix)**:
```python
# medusa_worker.py forward_batch_generation EXTEND branch
if batch.forward_mode.is_extend():
-    batch.spec_algorithm = SpeculativeAlgorithm.NONE   # drop
    model_worker_batch = batch.get_model_worker_batch()
    ...
```

Expected outcome: all 4 phases × all fields → 0 diffs. Documented in §5.4 EN+ZH, commit `ad45342dd`. fcloud paused. **Awaiting your "go" for iter 4.** 



