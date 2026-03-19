I’m tracing that exact DeepGEMM warning in the code so I can tell you whether it is on your active path or just a generic startup warning, and whether it can plausibly explain the accuracy drop you’re seeing.

Searched for regex `DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0|scale_fmt of checkpoint is not ue8m0|DeepGemm is enabled` (`**/python/**`), 4 results

Searched for regex `gptq_marlin|convertible to gptq_marlin during runtime` (`**/python/**`), 40 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/fp8_utils.py#300-300), lines 300 to 420

I found the warning source. It is tied to DeepGEMM/FP8 checkpoint handling, not directly to your GPTQ Marlin path, so I’m checking the exact condition in model config before answering how much weight to put on it.

Searched for regex `scale_fmt|ue8m0|deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM|DeepGEMM_SCALE_UE8M0` (`**/python/sglang/srt/**`), 50 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/configs/model_config.py#830-830), lines 830 to 870

I’m checking whether that warning is only advisory during model config load or whether it can actually affect the active kernels in your current MiniCPM GPTQ path, so I can tell you whether it’s a real suspect or just noise.

Searched for regex `DEEPGEMM_SCALE_UE8M0|ENABLE_JIT_DEEPGEMM|SGLANG_ENABLE_JIT_DEEPGEMM|fp8_gemm_runner_backend` (`**/python/sglang/srt/**`), 80 results

The warning path is broader than I expected: DeepGEMM is enabled globally by default on Blackwell, and the warning is emitted from model config parsing. I’m opening the DeepGEMM config module itself so I can tell you the cleanest way to disable it for an isolation run.

Searched for files matching `**/python/sglang/srt/layers/deep_gemm_wrapper/**`, 8 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py)

Read [](file:///home/oldzhu/sglang/python/sglang/srt/environ.py#300-300), lines 300 to 325

Yes, it could be related, and it is worth isolating.

What the warning actually means:
- In model_config.py, SGLang checks whether the checkpoint scale format is `ue8m0`.
- On Blackwell, configurer.py sets `DEEPGEMM_SCALE_UE8M0 = True` whenever DeepGEMM is enabled.
- If the checkpoint is not `ue8m0`, it emits exactly the warning you saw:
  - “This might cause accuracy degradation on Blackwell.”

So this is not a fake warning. The code explicitly treats this as a possible accuracy issue.

The important nuance is:
- your main quantized linear path is still `gptq_marlin`
- so DeepGEMM is probably not the main path for the GPTQ Marlin matmuls
- but DeepGEMM is enabled globally by default on Blackwell via environ.py
- and that means some FP8-related paths can still be affected even if you did not explicitly ask for DeepGEMM

So my answer is:

- Yes, this warning is a credible suspect.
- No, it does not prove DeepGEMM is the only cause.
- But it is now one of the cleanest things to isolate next, especially since your accuracy stayed low even after restoring the original sgl-kernel.

Why this matters to your current investigation:
- you already showed the accuracy drop persists even after reverting the sgl-kernel wheel
- that shifts suspicion away from Marlin kernel tuning
- this DeepGEMM warning is coming from the broader SGLang runtime path on Blackwell
- so it fits the new evidence much better than “our sgl-kernel changes broke correctness”

Also, seeing it 3 times at startup is not itself alarming. That usually just means model config / worker init is happening multiple times. The important part is the content of the warning, not the repetition.

The cleanest next isolation run is:
- keep the restored original sgl-kernel
- keep the same server args
- disable JIT DeepGEMM explicitly
- rerun correctness

Use this for the next run:
```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=0
python3 -m sglang.launch_server ...
```

Or if you want it inside the submission environment flow, temporarily add this export before launch in prepare_env.sh:
```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=0
```

What you should expect:
1. The DeepGEMM warning should disappear.
2. If `acc_ori` improves materially, then DeepGEMM-on-Blackwell with non-`ue8m0` scales was part of the problem.
3. If `acc_ori` does not improve, then this warning is not the main cause and we move on.

One more subtle point:
- `--fp8-gemm-backend triton` is not the best first isolation here
- the cleaner isolation is `SGLANG_ENABLE_JIT_DEEPGEMM=0`
- because the warning itself is controlled by global DeepGEMM enablement, not just your chosen FP8 GEMM backend flag

So the next best step is very clear:
1. original sgl-kernel
2. same launch args
3. `SGLANG_ENABLE_JIT_DEEPGEMM=0`
4. rerun correctness

If you want, after that run I can help you interpret the result and decide whether DeepGEMM should stay disabled for the next official submission candidate.
