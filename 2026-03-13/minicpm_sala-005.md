Based on the provided NVIDIA documentation for the upcoming competition, you are correct that the **RTX PRO GPU** uses the **SM120 architecture**, which natively supports **FP8 Tensor Cores** (yielding **296 TFLOPS**).

To utilize **FlashAttention-3 (FA3)** and its native FP8 support on this hardware, you must ensure your entire software stack—from the compiler to the attention backend—is targeting the `sm_120` compute capability.

### 1. Toolchain Compilation for SM120

The document explicitly states that you must **compile your toolchain based on sm120**. This involves several layers:

* **NVCC Compiler Flags:** When compiling CUDA kernels (including FlashInfer or FA3), you must pass the flag `-gencode arch=compute_120,code=sm_120`. This allows the compiler to use new SM120-specific features like **TMA (Tensor Memory Accelerator)** and **QMMA (mxfp8)**.


* **FlashAttention-3 (FA3) Requirements:** FA3 is designed for Hopper (SM90) and newer architectures. Because SM120 follows an **optimization strategy similar to Hopper**, FA3 is the intended native backend.


* **PyTorch/SGLang Build:** If you are building SGLang from source, ensure your environment variable `TORCH_CUDA_ARCH_LIST` includes `12.0` (or `9.0+` depending on the version's mapping) to ensure the JIT and AOT engines don't fall back to older `fa2` templates.

### 2. Key Changes in SM120 Kernel Programming

When adapting your code for this hardware, keep in mind these architectural specifics from the document:

* 
**MMA Granularity:** Unlike Hopper, where MMA (Matrix Multiply-Accumulate) commands are at the warpgroup level, in **sm120, they are at the warplevel**.


* 
**Hardware Features:** You must leverage **TMA for load/store operations** and **QMMA for mxfp8** to reach the advertised performance targets.


* 
**Memory Specs:** Your toolchain should be optimized for a **1398 GB/s memory bandwidth** and **84GB of GDDR7** VRAM.



### 3. Verification and Optimization Tools

The document recommends using the following tools to ensure your toolchain is actually utilizing the SM120 hardware effectively:

* 
**NVIDIA Nsight Systems:** Use this for system-wide performance analysis and to visualize the timeline of your CUDA kernels.


* 
**NVIDIA Nsight Compute:** Use this for a "deeper dive" into specific CUDA kernels to check **Occupancy**, **SM Utilization**, and **Instruction Throughput**.



### Recommended Next Step

To ensure your implementation is correct, NVIDIA recommends reviewing the **TensorRT-LLM reference usage** for FP8 block scale GEMM, which can be found in their GitHub unit tests (specifically lines 133–171).

Would you like me to help you draft the specific `cmake` or `setup.py` environment variables needed to force the SM120 build for your SGLang environment?
