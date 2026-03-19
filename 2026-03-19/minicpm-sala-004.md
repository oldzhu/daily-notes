I’m checking the existing UE8M0 requantization support in SGLang so I can answer the second question precisely: whether this repo already has a way to convert scales, and whether that applies to your MiniCPM GPTQ path or only to FP8 checkpoint families.

Searched for regex `should_deepgemm_weight_requant_ue8m0|requant_weight_ue8m0_inplace|scale_fmt="ue8m0"|use_scale_ue8m0` (`**/python/sglang/srt/models/**`), 28 results

Searched for regex `MiniCPM|minicpm` (`**/python/sglang/srt/models/**`), 80 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/model_loader/utils.py#120-120), lines 120 to 172

Here is the MiniCPM-SALA-specific matrix for your current serving path, plus the answer on `ue8m0`.

**Current Path Matrix**

Based on your current launch profile in prepare_env.sh and the code paths we inspected, the stack is roughly:

1. **Main quantized linear layers**
- Active path: `gptq_marlin`
- Primary implementation: Marlin / sgl-kernel
- Relevant files:
  - gptq.py
  - marlin_utils.py
  - gemm.py
- What it does:
  - runs the W4A16 GPTQ Marlin GEMM for the core linear layers
- Main suspect for speed:
  - yes
- Main suspect for the new DeepGEMM warning:
  - probably no, not directly

2. **Full attention backend**
- Active path: `--attention-backend minicpm_flashinfer`
- Relevant files:
  - model_runner.py
  - minicpm.py
- What it does:
  - drives the dense attention path for MiniCPM through the selected backend
- Main suspect for speed:
  - yes
- Main suspect for DeepGEMM warning:
  - not the most direct one

3. **Lightning / linear attention part of MiniCPM**
- Active path: model-specific hybrid path inside MiniCPM
- Relevant files:
  - minicpm.py
  - attention_registry.py
- What it does:
  - handles the linear-attention-style mixer layers in SALA
- Main suspect for speed:
  - yes, but harder to tune safely
- Main suspect for DeepGEMM warning:
  - unclear / probably secondary

4. **KV cache**
- Active path: `--kv-cache-dtype fp8_e5m2`
- Relevant files:
  - server_args.py
  - model_runner.py
- What it does:
  - stores KV cache in FP8
- Main suspect for speed:
  - yes, especially decode bandwidth
- Main suspect for accuracy:
  - yes, in principle
- Main suspect for DeepGEMM warning:
  - no, this is a different FP8-related topic

5. **FP8 GEMM helper/backend family**
- Controlled by:
  - `--fp8-gemm-backend`
  - `SGLANG_ENABLE_JIT_DEEPGEMM`
- Relevant files:
  - fp8_utils.py
  - configurer.py
  - model_config.py
- What it does:
  - chooses backends like DeepGEMM / FlashInfer / CUTLASS / Triton for blockwise FP8 GEMM-related paths
- Main suspect for the warning:
  - yes
- Main suspect for current correctness regression:
  - now a real suspect

So the short operational view is:

- **GPTQ Marlin** is your main quantized linear engine.
- **DeepGEMM** is a separate FP8 GEMM backend family that is globally enabled on Blackwell if available.
- The DeepGEMM warning is not about Marlin itself.
- That is why reverting sgl-kernel alone did not recover `acc_ori`.

**Best isolation toggles from here**
If you want clean isolation with minimal changes:

1. **Isolate DeepGEMM**
```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=0
```

2. **Keep the rest fixed**
- `--quantization gptq_marlin`
- `--kv-cache-dtype fp8_e5m2`
- `--attention-backend minicpm_flashinfer`
- `--force-dense-minicpm`

That tells you whether DeepGEMM-related FP8 handling is part of the accuracy issue.

**Can we make `scale_fmt` become `ue8m0`?**
Answer: **in principle yes for some FP8 checkpoint families, but probably not as a simple switch for your current MiniCPM GPTQ path.**

Why:

1. There is existing support in the repo for UE8M0 requantization logic.
- See utils.py
- And model-specific paths like:
  - longcat_flash.py
  - deepseek_v2.py

2. But that support is **model-family-specific**.
- The grep results show explicit UE8M0 requantization support for DeepSeek / LongCat style models.
- We did **not** find corresponding MiniCPM-specific UE8M0 requantization handling in minicpm.py.

3. Your current main path is **GPTQ Marlin**, not a native FP8 weight checkpoint path.
- So “make the checkpoint scale format `ue8m0`” may not even be the right framing for the main W4A16 weights.
- This warning is about DeepGEMM expecting a certain FP8 scale format on Blackwell.
- That is different from GPTQ Marlin weight representation.

So the practical answer is:

- **Yes, UE8M0 conversion is technically possible in this repo for some FP8 model families.**
- **No, it is not currently an out-of-the-box MiniCPM-SALA GPTQ feature.**
- For your current branch, the far cheaper and safer experiment is:
  - disable DeepGEMM first
  - verify whether `acc_ori` recovers
- Only if that clearly helps would it be worth discussing a new feature to add MiniCPM-specific UE8M0-compatible preprocessing or loader behavior.

**What I would do next**
1. Finish the current correctness run with `SGLANG_ENABLE_JIT_DEEPGEMM=0`.
2. If `acc_ori` improves clearly, treat DeepGEMM as a confirmed contributor.
3. Then choose between:
   - keeping DeepGEMM disabled for submission
   - or proposing a new feature to make the relevant FP8 path UE8M0-compatible, if that is even applicable to MiniCPM-SALA.

If you want, once you have the DeepGEMM-disabled correctness result, I can tell you whether the next step should be:
1. a no-code submission profile change, or
2. a new proposal for MiniCPM-specific UE8M0-compatible preprocessing / loader support.
