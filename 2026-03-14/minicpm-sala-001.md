Here is the comprehensive technical document designed for VS Code Copilot. It outlines the current bottleneck on SM120 (Blackwell/RTX PRO) hardware and the two distinct implementation paths.

---

# Technical Design: Enabling FP8 Attention on SM120 (Blackwell)

## 1. Problem Statement

The **RTX PRO (SM120)** architecture supports native FP8 compute and features the **Tensor Memory Accelerator (TMA)** and **QMMA**. However, current stable versions of `flash-attention` and `flashinfer` often route SM120 to the **FlashAttention-2 (FA2)** generic backend.

**The Issue:** The FA2 backend contains a hardcoded assertion that rejects FP8:
`assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"`
This results in a runtime crash when `--kv-cache-dtype fp8` is enabled in SGLang, despite the hardware being fully capable of 8-bit operations.

---

## 2. Comparison of Implementation Paths

| Feature | **Option 1: Native SM120 Build** | **Option 2: Mixed-Precision Fallback** |
| --- | --- | --- |
| **Logic** | Native FP8 Storage + Native FP8 Compute | FP8 Storage + Dequantize to FP16 Compute |
| **Performance** | **Maximum (Peak TFLOPS)** | **High (Bandwidth Bound)** |
| **Effort** | High (Requires Source Rebuild + Patches) | Low (Python-only logic patch) |
| **Hardware Use** | Uses SM120 QMMA & TMA | Uses Standard SM89/120 Tensor Cores |
| **Stability** | Experimental/Bleeding Edge | Highly Stable |

---

## 3. Option 1: Native SM120 Toolchain Rebuild

Use this for maximum performance in competition. This forces the use of **FlashAttention-3 (FA3)** or optimized SM120 kernels.

### Step A: Toolchain Configuration

Set environment variables to target the Blackwell architecture specifically during compilation.

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4  # Limits RAM usage during heavy CUDA compilation

```

### Step B: Flash-Attn Source Modification

1. **Clone & Submodules:**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && git submodule update --init --recursive

```


2. **Patch `setup.py`:** Add `sm_120` to the `cc_flag` list to ensure `nvcc` generates SM120 instructions.
3. **Build:**
```bash
pip install -e . --no-build-isolation

```



### Step C: SGLang Backend Patching

Modify the SGLang dispatcher to recognize SM120 as a high-performance target.
**File:** `python/sglang/srt/layers/attention/flashinfer_backend.py` (or utility router)

```python
# Change the backend selection logic
major, minor = torch.cuda.get_device_capability()
if major >= 9: # Now includes 9 (Hopper) and 12 (Blackwell)
    return "fa3" 

```

---

## 4. Option 2: Mixed-Precision Fallback (Storage Only)

Use this if you want the memory savings of FP8 without the complexity of CUDA kernel programming.

### Logic

The KV Cache is stored in **FP8** (saving 50% VRAM), but inside the attention kernel, the values are dequantized to **FP16** before the math operation.

### Implementation Steps

**File to Patch:** `python/sglang/srt/layers/attention/minicpm_backend.py` (or relevant attention backend).

1. **Locate the Forward Planning:** Find where `use_fp8_compute` is defined.
2. **Inject Hardware Check:**

```python
import torch
device_cap = torch.cuda.get_device_capability()

# If hardware is Blackwell but software isn't ready for native FP8 compute
if device_cap[0] == 12:
    # Keep KV Cache in FP8 for memory savings
    self.kv_cache_dtype = "fp8_e5m2" 
    # Force math to happen in FP16 to avoid the FA2 backend assertion error
    self.use_fp8_compute = False 

```

---

## 5. Implementation Roadmap for Copilot

1. **Verification:** Run `nvidia-smi` and `torch.cuda.get_device_capability()` to confirm SM120.
2. **Strategy Selection:**
* If `flash-attn` is installed via `.whl`, **Option 2** is the only immediate path.
* If a source directory is available, **Option 1** is preferred for the competition.


3. **Execution:**
* **Plan:** "Apply Python-level patch to SGLang to force `use_fp8_compute=False` when `capability[0] == 12`."
* **Implement:** Modify the backend initialization to override flags.



---

### Pro-Tip for Competition Benchmarking

The provided PDF notes that **SM120 MMA commands are warp-level** (vs Hopper's warpgroup-level). If implementing **Option 1**, ensure your custom kernels or `fa3` templates are utilizing the **TMA** for asynchronous data loading to maximize the **1398 GB/s** memory bandwidth.

