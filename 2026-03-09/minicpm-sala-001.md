You’re partly right about the diagnosis, but the conclusion needs one correction.

sglang.launch_server --attention-backend minicpm_flashinfer works without flash_attn because SGLang runtime is using its own attention backend path.
preprocess_model.py (GPTQModel path) is not using SGLang runtime backend flags. It loads model via transformers + custom MiniCPM code first.
In that load path, MiniCPM-SALA currently requires flash_attention_2 and then transformers checks for flash_attn package. That is why quant-prep fails.
So:

--attention-backend minicpm_flashinfer is valid for serving, but it does not solve GPTQModel preprocessing load.
For current GPTQModel-based preprocess, you still need either:
flash_attn available in quantization environment, or
a different quantization path/tool that does not trigger this HF FA2 check.
Given your workflow, the practical route is:

run GPTQ preprocess in a compatible quant env (with flash_attn), output quantized model dir,
then serve/eval with your current SGLang minicpm_flashinfer backend.
If you want, I can propose the next single feature to support this cleanly: add a dedicated “external quant env handoff” mode in preprocess_model.py and docs so your submission flow stays reproducible.


