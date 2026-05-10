Compacted conversation

Searched for regex `dense_len|dense_as_sparse|is_sparse_layer` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#440-440), lines 440 to 485

Yes to the first part, partially yes to the second:

**1. Official default = dense for everything.**
`--attention-backend flashinfer` selects the stock FlashInfer backend, NOT `minicpm_flashinfer`. The custom MiniCPM backend is the *only* file that contains sparse code (top-k scoring, compress_k1/k2, sparse FA). With FlashInfer selected, sparse code is never even loaded — every layer (including the 8 sparse_attention layers) runs as dense full attention via FlashInfer. So the official toolkit default is fully dense, regardless of `--force-dense-minicpm`.

**2. With `minicpm_flashinfer` active and no `--dense-as-sparse`:**
`self.dense_len = hf_config.sparse_dense_len` (non-zero threshold from the model config). Then per-request, in `forward_batch.seq_lens_cpu`:
- `seq_len < dense_len` → that request takes the **dense branch inside minicpm_backend.py**, which calls FlashAttention (FA varlen) — *not* the FlashInfer backend. It's "full attention", but the kernel is FA inside the MiniCPM backend, not flashinfer. (`minicpm_flashattn` variant uses the same FA path; `minicpm_flashinfer` variant routes the dense branch through FlashInfer's prefill/decode — depends on which sub-backend.)
- `seq_len >= dense_len` → sparse branch (top-k + sparse FA).

So in a mixed batch under `minicpm_flashinfer`, short requests get dense attention and long requests get sparse — and `--dense-as-sparse` only forces `dense_len=0` so everything (even short batches) goes through the sparse path. That's why `--dense-as-sparse` widens sparse coverage rather than activating it.

**Implication for our Round 13e Test 1:** we ran `minicpm_flashinfer` + `--dense-as-sparse` + bs=8 at long context — i.e. we forced *every* request through the slow sparse decode path, including ones that would have been short enough to take the cheap dense branch. That's a contributor to the timeouts on top of the inherent sparse-decode cost.

This is worth noting before option B profiling: we may want a profile run **without** `--dense-as-sparse` to isolate sparse-only-on-long-seq cost from forced-sparse-on-short-seq cost. Want me to fold that into the CHANGE_0135 profiling proposal? 


