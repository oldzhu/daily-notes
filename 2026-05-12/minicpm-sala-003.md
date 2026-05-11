Let us document the bug and the change 0137 try to fix it for our reference and tracking - I am curious why we didn't see the off-by-one in previous sparse testing why we start to see the issue after the change 0136. 
then let us temporarily Park CHANGE_0136 and pivot and dscuss more on the 3f-1 (flashinfer), besides a little bit accuracy loss, we did see some improving in speed benchmark, if we can make the accuracy improving to near 80% and it can be the next baseline of the submission and improving target. I would like to understand the difference bewteen flashinfer backend and minicpm-flashinfer backend, r u sure if we set the backend to flashinfer, then it can't handle minicpm mixed layers(sparse and lighting) or at least they are not compatible? could u list what real code flow difefrerence  using flashinfer and minicpm-flashinfer?
=======

Created 5 todos

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#1050-1050), lines 1050 to 1110

Searched for regex `sparse_page_table\s*=|sparse_page_table.shape` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 12 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#410-410), lines 410 to 485

Searched for regex `sparse_page_table|build_sparse_prefill_metadata` (`**/python/sglang/srt/layers/attention/minicpm_sparse_utils.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_sparse_utils.py#1395-1395), lines 1395 to 1470

Found the off-by-one. Let me check the flashinfer vs minicpm_flashinfer distinction. 

Searched for regex `minicpm_flashinfer|MiniCPMAttentionBackend|force_dense_minicpm` (`**/python/sglang/srt`), no results

Searched for regex `minicpm_flashinfer|force_dense_minicpm` (`**/python/sglang/srt/**`), 37 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1515-1515), lines 1515 to 1530

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/attention_registry.py#160-160), lines 160 to 220

Searched for regex `layer_id|sparse_layer|dense_layer|forward_extend|forward_decode` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 8 results

Searched for regex `RadixAttention|self.attn|sparse|hybrid|layer_type|is_sparse` (`**/python/sglang/srt/models/minicpm.py`), 30 results

Searched for regex `full_attn_backend|attention_backend|MiniCPMSparseBackend|minicpm` (`**/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`), 30 results

I have enough to write the docs. Let me commit them. 

Starting: *Write CHANGE_0137 EN/ZH docs* (4/6)

Searched for files matching `**/docs/soar_2026_changes/CHANGE_0136*`, 2 matches

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0137_sparse_prefill_page_table_off_by_one.zh.md)

Now the flashinfer-vs-minicpm_flashinfer comparison doc and parking note for CHANGE_0136.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md)

Now park CHANGE_0136 with a status note appended at top of both EN/ZH docs: 

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0136_minicpm_sparse_dense_len_flag.en.md#1-1), lines 1 to 15

Replacing 4 lines with 14 lines in [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0136_minicpm_sparse_dense_len_flag.en.md), [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md)


```

```

```

```


Ran terminal command:  cd /home/oldzhu/sglang && git add docs/soar_2026_changes/CHANGE_0136_minicpm_sparse_dense_len_flag.en.md docs/soar_2026_changes/CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md docs/soar_2026_changes/CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md docs/soar_2026_changes/CHANGE_0137_sparse_prefill_page_table_off_by_one.zh.md docs/soar_2026_changes/RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md docs/soar_2026_changes/RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md && git commit -m "docs: park CHANGE_0136, file CHANGE_0137 (sparse prefill off-by-one), and RESEARCH on flashinfer vs minicpm_flashinfer code flow" 2>&1 | tail -5 && git push minicpm-src mixed_minicpm_cudagraph 2>&1 | tail -5

Pushed as `ae18ed8a5`. Three new doc pairs landed:

## CHANGE_0137 — sparse-prefill off-by-one

**Root cause**: allocator at `minicpm_sparse_utils.py:1421` sizes
`sparse_page_table.shape[1]` by `extend_seq_lens_cpu[i]` (chunk length, 103),
but writer at `minicpm_backend.py:1086` slices with `seq_lens_cpu[dense_bs]`
(full kv length, 104 due to overlap-mode `+1` pre-update).

**Why it stayed hidden until CHANGE_0136**:
- Test 12 / submission baseline → `--force-dense-minicpm` rewrites the backend to stock flashinfer; `MiniCPMSparseBackend` never instantiated.
- `--dense-as-sparse` baselines → forces `dense_len=0`, every request takes the wide-row branch, dense_bs_list always empty.
- `SOAR_SPARSE_MODE=1` (Round 13d) → bug WAS reachable but boot-time `current_seed` crash (now fixed by CHANGE_0133's compile-disable) prevented us from ever reaching forward_extend on a real request.
- CHANGE_0136 with high `dense_len` is the first config that puts **all** requests through the narrow else-branch + dense_bs writer at once → instant repro.

**Proposed fix**: size by `int(forward_batch.seq_lens_cpu[i])` in the else branch (one-line change, dropped writer-side workarounds explicitly).

## RESEARCH on flashinfer vs minicpm_flashinfer

Key finding answering your direct question: **stock `flashinfer` is functionally compatible with SALA's mixed layers**. Lightning layers always go through `SimpleGLAAttnBackend` regardless. Sparse std-attn layers under stock flashinfer just compute full dense attention (a superset of top-k). It will not crash and it produces valid outputs.

**The 2.4-pt accuracy delta is not from kernel choice** — Test 12 (`force_dense_minicpm + minicpm_flashinfer`) and Round 13f-1 (no `force_dense_minicpm + flashinfer`) **run the same stock FlashInfer kernel** because `server_args.py:1525` rewrites `minicpm_flashinfer→flashinfer` whenever `force_dense_minicpm=True`. The accuracy gap therefore must come from peripheral knobs that flip with `force_dense_minicpm`:

1. **Lightning-mixer `recurrent_threshold`** — 64 (no force) vs 128 (force). Most likely culprit. Many eval prompts land in [64,128) and switch from exact-recurrent to chunked SimpleGLA → low-precision drift.
2. `model_config.has_sparse_attention` True ↔ False — secondary side-effects.
3. `--dense-as-sparse` / compile flags removed in 13f-1 — minor.

**Recommended next experiments** (cheapest first):
- **Exp B**: Path B + `--force-dense-minicpm` (which is just Test 12 itself but reached via the explicit `flashinfer` string). If acc returns to 79.29% and speed stays at ~110/40/33 → that's already the next baseline. If speed regresses to 121/44/35 → the 13f-1 speed gain came from dropping force-dense, and we now know the cost.
- **Exp A**: Path B + `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128`. Cheapest knob isolation for hypothesis 1.

CHANGE_0136 docs both EN/ZH now carry a "PARKED" header pointing to CHANGE_0137 + this research doc.

Want me to draft the prepare_env.sh / scripts changes for Exp A and Exp B as a proposal so you can decide which to run first when fcloud is back up? 



