Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_NVFP4_KV_CACHE_20260427_1730.en.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)

I'll work on both in parallel: NVFP4 KV P1 survey (research, offline) and W4-FP8 spike code (write the .cu file, leave fcloud execution for later approval).   

Starting: *NVFP4 KV P1: survey sglang KV dtype plumbing* (1/5)

Searched for regex `nvfp4|NVFP4|fp4|FP4|e2m1` (`**/python/sglang/srt/**`), 80 results

Searched for regex `kv_cache_dtype|kv-cache-dtype|fp8_e5m2|fp8_e4m3` (`**/python/sglang/srt/**`), 80 results

Searched for files matching `**/kvfp4_tensor*`, 2 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#170-170), lines 170 to 220

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/attention/minicpm_backend.py#820-820), lines 820 to 980

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#1080-1080), lines 1080 to 1210

Searched for regex `forward_decode|set_kv_buffer|get_kv_buffer|k_cache|v_cache|cache_k|cache_v` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/kvfp4_tensor.py)

I have enough survey data. Let me write the P1 result doc and the spike scaffold.  

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SURVEY_NVFP4_KV_P1_20260428_1130.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)

 

Starting: *NVFP4 KV P1: identify gap and draft P2 plan* (2/5)

Starting: *W4-FP8 spike: author bench_w4fp8_sm120.cu* (3/5)

Now the W4-FP8 spike. I'll write a self-contained CUDA benchmark plus build/run script, and a result template. 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/spike_w4fp8/bench_w4fp8_sm120.cu)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/spike_w4fp8/run_w4fp8_spike.sh)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md)

 

Starting: *Document P1 results + commit* (5/5)

Ran terminal command:  cd /home/oldzhu/sglang && chmod +x docs/soar_2026_changes/spike_w4fp8/run_w4fp8_spike.sh && git add docs/soar_2026_changes/SURVEY_NVFP4_KV_P1_20260428_1130.en.md docs/soar_2026_changes/SURVEY_NVFP4_KV_P1_20260428_1130.zh.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md docs/soar_2026_changes/RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md docs/soar_2026_changes/spike_w4fp8/ docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.en.md docs/soar_2026_changes/chat/CHAT_nvfp4-survey-w4fp8-spike_20260428_1200.zh.md && git commit -m "docs: NVFP4 KV P1 survey + W4-FP8 spike scaffold (R10)" && git push minicpm-src mixed_minicpm_cudagraph

Completed: *Document P1 results + commit* (5/5)

Both tasks delivered offline (no fcloud burned). Committed `f34064012` to `minicpm-src/mixed_minicpm_cudagraph`.

**1. NVFP4 KV P1 survey** — SURVEY_NVFP4_KV_P1_20260428_1130.en.md / zh
- Tree already has ~80% of plumbing (server arg, dtype config, `MHATokenToKVPoolFP4`, `KVFP4QuantizeUtil`).
- **Important labeling note**: sglang's `fp4_e2m1` is actually **MXFP4** (16-elem block + e8m0/uint8 scale), not NVFP4. For our purposes that's fine and even saves more memory.
- Real gaps are **all in minicpm_backend.py** (4 gates and one `set_kv_buffer` k_scale interaction).
- Memory: **~44% smaller than FP8**.
- Updated P2 plan: ~3 days, starts with `--force-dense-minicpm` smoke test to isolate plumbing from sparse gates.
- **3 questions for you before P2**: PyTorch ≥2.8 on fcloud? Approve dense-only smoke first? Accuracy regression tolerance?

**2. W4-FP8 spike scaffold** — ready to upload to fcloud
- spike_w4fp8/bench_w4fp8_sm120.cu — hand-rolled W4 → FP8 dequant + inline-PTX `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Times 100 iters, prints TFLOPS + GREEN/YELLOW/RED verdict vs 281 TF FP8 ceiling.
- spike_w4fp8/run_w4fp8_spike.sh — builds `nvcc -arch=sm_120 -O3` and runs 4 shape variants.
- RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md — result template with decision tree.

**Awaiting your approval** to upload + run the spike on fcloud (~1 fcloud-hour, no production impact). Will not touch fcloud without explicit go-ahead per the cost-saving rule. 



