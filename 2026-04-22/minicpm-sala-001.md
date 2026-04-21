Starting: *Document EAGLE3 status & pause* (1/3)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#1-1), lines 1 to 150

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/eagle3_implementation_plan.md)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md#150-150), lines 150 to 200

Now let me check the current best test results and understand NVFP4 KV cache feasibility: 

Research NVFP4 KV cache support

Ran terminal command:  ls -la /home/oldzhu/sglang-nvfp4/ 2>/dev/null && echo "---EXISTS---" || echo "---NOT FOUND---"

Searched for text `kv_cache_dtype` (`**/python/sglang/srt/server_args.py`), 20 results

Searched for regex `nvfp4|fp4|nv_fp4|fp4_e2m1` (`**/python/sglang/**`), 20 results

Searched for text `kv_cache_dtype` (`**/python/sglang/srt/model_executor/model_runner.py`), 20 results

Searched for regex `fp4|nvfp4` (`**/sgl-kernel/**`), 20 results

Ran terminal command:  cd /home/oldzhu/sglang-nvfp4 && git log --oneline -20 && echo "---BRANCH---" && git branch --show-current

Searched for regex `fp4_e2m1|fp4|float4` (`**/python/sglang/srt/server_args.py`), 20 results

Searched for files matching `**/kvfp4*`, 2 matches

Searched for regex `mixed.*kv.*cache|per.*layer.*kv|kv.*dtype.*layer`, 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_executor/model_runner.py#1527-1527), lines 1527 to 1575

Ran terminal command:  cd /home/oldzhu/sglang-nvfp4 && git diff --stat minicpm-src/mixed_minicpm_cudagraph..nvfp4-weight-quant | tail -20

Searched for regex `fp4_e2m1|float4_e2m1|kvfp4|fp4.*kv|kv.*fp4` (`**/python/sglang/srt/layers/attention/**`), 4 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1670-1670), lines 1670 to 1790

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/kvfp4_tensor.py#1-1), lines 1 to 80

Ran terminal command:  cd /home/oldzhu/sglang-nvfp4 && head -50 benchmark/soar/demo_sala/quantize_nvfp4.py

Searched for regex `fp4_e2m1|float4_e2m1|kv4|FP4` (`**/python/sglang/srt/layers/attention/**`), 10 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/server_args.py#1775-1775), lines 1775 to 1870

Searched for regex `fp4|float4|kv4` (`**/python/sglang/srt/layers/attention/flashattention_backend.py`), no results

Searched for regex `fp4|float4|kv4` (`**/python/sglang/srt/layers/attention/triton_backend.py`), no results

Searched for regex `fp4|float4|kv4` (`**/python/sglang/srt/layers/attention/flashinfer_backend.py`), no results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/kvfp4_tensor.py#75-75), lines 75 to 113

Searched for regex `fp4|float4|kv4|kv_cache_dtype` (`**/python/sglang/srt/layers/attention/minicpm_backend.py`), 20 results

Searched for regex `fp4|float4|kv_cache_dtype` (`**/python/sglang/srt/mem_cache/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/mem_cache/memory_pool.py#1080-1080), lines 1080 to 1200

Updated todo list

Now let me create the EAGLE3 documentation and update memory: 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0090_eagle3_speculative_decoding.en.md)

 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/CHANGE_0090_eagle3_speculative_decoding.zh.md)

Starting: *Save EAGLE3 memory for later* (2/3)

Read memory

Created memory file [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/repo/eagle3_paused_status.md)

Starting: *Analyze next optimization directions* (3/3)

Now let me check the NVFP4 KV cache research and toolkit page for latest guidance:   



Fetched https://soar.openbmb.cn/toolkit

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/nvfp4_kv_cache_research.md)

Read memory [](file:///home/oldzhu/.vscode-server/data/User/workspaceStorage/2dc4bd7000a584b7a90a70c28bdf3afd/GitHub.copilot-chat/memory-tool/memories/YzZjNWYyNDctMDQ0ZS00MmRkLThiNzctZWU2N2QzM2ZmZGY1/fp4_kv_cache_hold.md)

Completed: *Analyze next optimization directions* (3/3)

EAGLE3 is documented and saved. Now for next directions:

---

## Next Optimization Directions Analysis

### Current Position
- **Baseline best (Test 20)**: S1=113.67s, S8=41.07s, Smax=34.15s, accuracy=80.64%, C=1.0
- **Score**: 56.63 (#19), target ≥79.55 (#5) → need **~40% faster** across all tiers
- **Already applied**: torch.compile(max-bs=8), mixed-chunk, max-running-req=24, GPTQ+FP8+dense

### Your idea: GPTQ + Mixed NVFP4 KV Cache + Dense

**Concept**: Replace FP8 KV cache with NVFP4 (FP4 E2M1) for middle layers while keeping FP8 for sensitive first/last layers.

| Aspect | Assessment |
|--------|-----------|
| **Speed gain** | FP4 = 2× less memory bandwidth vs FP8 → faster attention decode |
| **Accuracy risk** | Pure FP4: champion saw 80%→75% (C=0). Mixed: recoverable to ~near-FP8 |
| **Implementation** | **HIGH effort** — sglang has NO per-layer mixed KV cache dtype. Need new memory pool class, per-layer dtype routing in minicpm_backend |
| **Hardware** | `torch.float4_e2m1fn_x2` needs PyTorch 2.8+ on Blackwell. fcloud has RTX 6000D (Blackwell sm_120) — should work |
| **Scope** | Only affects 8 standard attention layers (24 SimpleGLA layers use recurrent state, no KV cache) |

**Verdict**: The FP4 KV cache only benefits 8/32 layers. The 24 SimpleGLA layers (75% of compute) are unaffected. The implementation cost is high for limited scope.

### All Viable Options (ranked by expected ROI)

| # | Direction | Expected Gain | Effort | Risk | Notes |
|---|-----------|--------------|--------|------|-------|
| **1** | **SimpleGLA state contiguity (A1)** | 5-8% decode | Medium | Low | Ensure fast I/O path always triggers for 24 recurrent layers. Highest ROI — affects 75% of compute |
| **2** | **Fused RMSNorm + residual_scale (K4)** | 1-2% | Medium | Low | Save 64 kernel launches per forward. Custom CUDA kernel mod |
| **3** | **Residual scale folding (M1)** | 1-3% | Low | Zero | Fold constant scalar into weights at load time. Mathematically equivalent |
| **4** | **Fuse state I/O into FLA kernel (A3)** | 10-15% decode | High | Medium | Eliminate state load/store per decode step for 24 layers. **Highest potential** but requires FLA CUDA kernel mods |
| **5** | **Mixed NVFP4 KV cache** | 2-4% | High | Medium | Your idea. Only helps 8 attention layers. Needs per-layer dtype + backend changes |
| **6** | **Workspace buffer pooling (G2)** | 0.5-1% | Low | Zero | Reuse Marlin GEMM workspace across layers |
| **7** | **NVFP4 weight quantization** | 5-15%? | Medium | High | W4→FP4 weights with CUTLASS FP4 GEMM on Blackwell. Tested previously — server crashed (nvfp4 branch). Needs investigation |
| **8** | **Dense GPTQ recalibration (G3)** | 2-5% | Very High | Medium | Re-quantize weights calibrated for dense attention. Hours of calibration |
| **9** | **Custom Triton decode kernel for SimpleGLA** | 10-20% | Very High | High | Write optimized single-step recurrent kernel fused with state I/O |

### My Recommendation

**Priority order**:
1. **M1 (residual scale folding)** — zero risk, easy, 1-3%
2. **A1 (SimpleGLA state contiguity)** — medium effort, 5-8% on the dominant compute path
3. **NVFP4 weight quantization (#7)** — revisit the crashed nvfp4 branch, Blackwell has native FP4 tensor cores → could be a big win if we can make it work
4. **K4 (fused RMSNorm+scale)** — medium effort, reliable 1-2%

The **mixed NVFP4 KV cache** is possible but has limited ROI since only 8/32 layers use KV cache. If you're drawn to the FP4 direction, **NVFP4 weight quantization** (replacing GPTQ Marlin with CUTLASS FP4 GEMM) would affect ALL layers and has higher potential on the Blackwell GPU we have. 

Which direction interests you most?

