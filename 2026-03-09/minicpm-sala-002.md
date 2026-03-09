Short answer: it can work, but it is not guaranteed unless format compatibility is verified.

Why:

preprocess_model.py (GPTQModel) and SGLang runtime are different stages.
Preprocess stage: creates quantized weights + metadata (quantize_config.json, packed tensors, etc.).
Runtime stage: SGLang loader reads those artifacts and maps them to its quant kernels (gptq / gptq_marlin) while attention execution still uses minicpm_flashinfer.
Quantized linear kernels and attention backend are mostly orthogonal.
GPTQ/GPTQ-Marlin affects linear/matmul weight path.
minicpm_flashinfer affects attention path.
So in principle, quantized weights + minicpm_flashinfer can coexist.
What can still break:
Quant format mismatch (GPTQModel output layout/version vs what SGLang expects).
Missing/incorrect quantize_config.json.
Unsupported quant method for this model shape/path.
Runtime memory/accuracy regressions.
Why success is plausible here:
SGLang has explicit GPTQ/GPTQ-Marlin support in loader/quantization modules.
Your runtime launch can explicitly select quantization (--quantization gptq or gptq_marlin) independent of attention backend.
So no fundamental architectural conflict, but you must validate end-to-end.

Minimal proof checklist after quantization:

Quantized dir contains quantize_config.json.
Server starts with:
--model-path <quant_dir>
--quantization gptq (or gptq_marlin if artifact matches)
--attention-backend minicpm_flashinfer
One chat request succeeds.
Then run correctness + 3x speed.
If you want, I can give you a strict A/B validation matrix (startup, functional, correctness, speed, rollback) in one compact command list.

