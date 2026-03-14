In the SGLang codebase, the `minicpm_flashinfer` backend routing is primarily handled in the `minicpm_backend.py` file. While it doesn't always say "flash_attention_2" as a literal string in the way you might expect, it sets up the **FlashInfer** wrapper which, on your hardware, automatically selects the `fa2` (FlashAttention-2) templates.

You can find the relevant logic in your local repository at:
`python/sglang/srt/layers/attention/minicpm_backend.py`

### Specific Areas to Check:

**1. The `ForwardMode` and Backend Initialization:**
Look for the `MiniCPMAttentionBackend` class. Inside the `__init__` or the metadata capture methods, it initializes the `FlashInfer` wrapper.

```python
# Look for where the wrapper is instantiated, usually something like:
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

```

If you see the code explicitly selecting a backend or if it's defaulting to `flashinfer` while your hardware is detected as SM 12.0, the "Selection" happens inside the FlashInfer library's `plan` or `begin_forward` methods.

**2. The `begin_forward` Call:**
Near line **1640** (based on your error log), you will see the call to `flashinfer_wrapper.begin_forward`.
This is the point where the software decides which CUDA kernel to JIT-compile. Even if it doesn't say "FA2" here, the `flashinfer` library receives your hardware capability (12.0), realizes it doesn't have a specific `sm120` native FP8 path, and routes you to the generic `fa2` backend.

**3. Hardcoded Implementation Strings:**
If you have a custom or older fork of SGLang for MiniCPM, check for this specific variable:

```python
# Search for this variable in minicpm_backend.py
self.backend_type = "flash_attention_2" 

```

If this is set, it bypasses the FlashInfer auto-selection and forces the legacy FA2 path which lacks FP8 support.

### How to Find it Quickly:

Run the following command in your `sglang` root directory to find exactly where the backend is being specified:

```bash
grep -rnE "flash_attention_2|minicpm_flashinfer" python/sglang/srt/layers/attention/

```

### Recommendation for your Copilot Doc:

When you use the document I generated, tell Copilot:

> "Search `python/sglang/srt/layers/attention/minicpm_backend.py` for the `FlashInfer` wrapper initialization. Locate where it handles the `dtype` and `use_fp8_compute` flags, and apply the Step C patch to ensure it doesn't fall back to the FA2 assertion error."
