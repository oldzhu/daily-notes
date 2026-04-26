Compacted conversation

Created 4 todos

Starting: *Document Option B vs C comparison* (1/4)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes)

Let me create the B vs C comparison docs and then investigate the Option B integration details in parallel.  

Read [](file:///home/oldzhu/sglang/sgl-kernel)

Read [](file:///home/oldzhu/sglang/docs/soar_2026_changes/DECISION_fp8_w4_implementation_options.md#1-1), lines 1 to 60

Ran terminal command:  ls /home/oldzhu/sglang/sgl-kernel/csrc/ && ls /home/oldzhu/sglang/sgl-kernel/cmake/ && head -80 /home/oldzhu/sglang/sgl-kernel/CMakeLists.txt

Ran terminal command:  ls /home/oldzhu/sglang/sgl-kernel/csrc/gemm/ && ls /home/oldzhu/sglang/sgl-kernel/csrc/cutlass_extensions/ 2>/dev/null | head -30

Ran terminal command:  ls /home/oldzhu/sglang/sgl-kernel/csrc/gemm/gptq/ && ls /home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/

Ran terminal command:  cat /home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu | head -80

Ran terminal command:  grep -r "sm100\|sm_100\|Blackwell\|blackwell\|SM120\|sm120" /home/oldzhu/sglang/sgl-kernel/csrc/gemm/ --include="*.cu" --include="*.cuh" -l && grep -r "fp8_blockwise_gemm\|blockwise_gemm" /home/oldzhu/sglang/sgl-kernel/python/ -r | head -20

Ran terminal command:  grep -r "fp8_blockwise\|blockwise_scaled_mm\|scaled_fp8_mm" /home/oldzhu/sglang/sgl-kernel/python/ -r | head -30 && grep -r "fp8_blockwise\|blockwise_scaled_mm\|scaled_fp8_mm" /home/oldzhu/sglang/python/sglang/ -r | grep -v ".pyc" | head -20

Ran terminal command:  cat /home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu | grep -A 20 "void fp8_blockwise_scaled_mm\|void launch_sm100\|dispatch" | head -60 && grep -n "arch_check\|SM100\|sm100\|is_sm" /home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu | head -20

Good - I can see sgl-kernel already has full CUTLASS 3.x infrastructure with SM100/Blackwell FP8 GEMM! Now let me create the comparison doc and start the Option B proposal. 

Updated todo list

Ran terminal command:  tail -100 /home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu

This is a major discovery — sgl-kernel **already has SM120 FP8 blockwise GEMM** (`sm120_fp8_blockwise_dispatch_shape`)! Option B is much simpler than expected. Let me now create the comparison docs and the proposal. 

Ran terminal command:  grep -n "sm120_fp8_blockwise\|SM120" /home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu | head -20 && find /home/oldzhu/sglang/sgl-kernel/csrc -name "*sm120*" -o -name "*SM120*" 2>/dev/null

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu#206-206), lines 206 to 390

Now I have the complete picture. Let me create both the comparison docs and the Option B proposal. 

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/DECISION_option_b_vs_c_deep_comparison.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/DECISION_option_b_vs_c_deep_comparison.zh.md)

Starting: *Create Option B implementation proposal* (2/4)

Now let me look at the preprocess_model.py and linear.py to understand the integration points for the proposal. 

Ran terminal command:  grep -n "gptq\|marlin\|w4\|W4\|quantize\|dequant" /home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py | head -40 && wc -l /home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py

Ran terminal command:  grep -n "def run_\|def copy_\|quant_mode\|SOAR_QUANT_MODE\|def main\|def _run" /home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py | head -30

Ran terminal command:  grep -n "FP8\|fp8\|blockwise\|scaled_mm" /home/oldzhu/sglang/python/sglang/srt/layers/linear.py | head -30 && grep -n "class.*Linear\|def forward" /home/oldzhu/sglang/python/sglang/srt/layers/linear.py | head -30

Ran terminal command:  grep -n "MarlinLinear\|GPTQMarlin\|marlin\|gptq_marlin\|QuantLinear\|quant_method" /home/oldzhu/sglang/python/sglang/srt/layers/linear.py | head -20

Now let me create the Option B implementation proposal: 

Ran terminal command:  grep -n "SOAR_QUANT_MODE\|mode.*copy\|mode.*gptq\|def main\|argparse\|choices" /home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py | head -20

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/preprocess_model.py#1085-1085), lines 1085 to 1120

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_option_b_fp8_blockwise_gemm.en.md)

Created [](file:///home/oldzhu/sglang/docs/soar_2026_changes/PROPOSAL_option_b_fp8_blockwise_gemm.zh.md)

Starting: *Push all docs to minicpm-src* (3/3)

Ran terminal command:  cd /home/oldzhu/sglang && git add \
  docs/soar_2026_changes/DECISION_option_b_vs_c_deep_comparison.en.md \
  docs/soar_2026_changes/DECISION_option_b_vs_c_deep_comparison.zh.md \
  docs/soar_2026_changes/PROPOSAL_option_b_fp8_blockwise_gemm.en.md \
  docs/soar_2026_changes/PROPOSAL_option_b_fp8_blockwise_gemm.zh.md && \
git commit -m "docs: Option B vs C deep comparison + Option B FP8 blockwise GEMM proposal

Key finding: sgl-kernel ALREADY has SM120 FP8 blockwise GEMM:
  sm120_fp8_blockwise_dispatch_shape() in fp8_blockwise_gemm_kernel.cu
  Guard: CUTLASS_ARCH_MMA_SM120A_SUPPORTED / CUTLASS_ARCH_MMA_SM120_SUPPORTED
  Python API: fp8_blockwise_scaled_mm() already exposed

This means Option B requires ZERO CUDA kernel writing.

DECISION_option_b_vs_c_deep_comparison.en/zh.md:
- Head-to-head comparison across all inference stages (S1/S8/Smax)
- Option B: no CUDA, 3-5 days, 2x weight memory, ~2.5-3.5x prefill TFLOPS
- Option C: 2000-3000 lines CUDA, 4-8 weeks, W4 memory kept, ~3-4x TFLOPS
- Decode trade-off: B has 2x weight BW cost; C has same BW as baseline
- Option C reference implementation notes preserved for future

PROPOSAL_option_b_fp8_blockwise_gemm.en/zh.md:
- Day-by-day 5-day implementation plan
- preprocess_model.py: new fp8_blockwise mode (FP16 -> FP8 blockwise)
- New FP8BlockwiseLinearMethod quantization class
- Scale format derivation: scales_b shape (N/128, K/128) = (K_dim/128, N_dim/128)
- Validation commands and expected results table
- Risk assessment + rollback plan" && \
git push minicpm-src mixed_minicpm_cudagraph

Completed: *Push all docs to minicpm-src* (3/3)

All docs committed (b07f135d8) and pushed to `minicpm-src`. Here's a summary of what was done and what's ready:

---

**Documentation created and pushed**:

1. DECISION_option_b_vs_c_deep_comparison.en.md / .zh.md — Complete B vs C reference including Option C implementation guide for future use

2. PROPOSAL_option_b_fp8_blockwise_gemm.en.md / .zh.md — Day-by-day implementation plan for Option B

---

**Key discovery that changes everything**: sgl-kernel already has `sm120_fp8_blockwise_dispatch_shape()` — **no CUDA kernel writing needed**. Option B is pure Python + PyTorch.

**Option B implementation plan (5 days)**:
- **Day 1**: Add `fp8_blockwise` mode to preprocess_model.py (FP16 → FP8 blockwise 128×128 blocks)
- **Day 2**: New `FP8BlockwiseLinearMethod` class + activation quantizer
- **Day 3**: Wire into `minicpm.py` + server integration
- **Day 4**: Accuracy validation on fcloud (target >99% normalized)
- **Day 5**: Speed benchmark (S1/S8/Smax)

**Option C** is preserved with full implementation notes in the comparison doc — activate if B's 30-50% speedup still falls short of top-5 gap, or if W4 memory efficiency becomes critical.

Ready to proceed with Day 1 implementation upon your approval to start coding. 



