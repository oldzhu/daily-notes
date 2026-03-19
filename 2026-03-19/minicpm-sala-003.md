DeepGEMM is a specialized GEMM library focused on very fast modern NVIDIA GPU matrix multiplication, especially FP8-style paths on Hopper and Blackwell. In SGLang, it is not the universal matrix-multiply backend for everything. It is one possible backend that some quantized or FP8 execution paths can use when the hardware and tensor format fit.

In your repo, the relevant places are:
fp8_utils.py
configurer.py
model_config.py

What DeepGEMM is used for

In this codebase, DeepGEMM is mainly used as a backend for blockwise FP8 GEMM-related paths, not as the primary backend for your GPTQ Marlin W4A16 linear path.

You can see that in fp8_utils.py, where SGLang chooses among FP8 GEMM runners such as:
- DeepGEMM
- FlashInfer TRTLLM
- CUTLASS
- Triton
- AITER

That means:
- if a path needs FP8 block GEMM, DeepGEMM may be selected
- if a path is GPTQ Marlin, that main path is different and uses the Marlin kernel family instead

So for your current MiniCPM-SALA setup:
- DeepGEMM is probably not the main engine for GPTQ Marlin linear layers
- but it can still affect some FP8-related execution paths on Blackwell
- and the startup warning you saw means SGLang believes DeepGEMM is active enough to matter

Why DeepGEMM exists

The reason DeepGEMM exists is simple:
- very new GPUs like Hopper and Blackwell have hardware features that generic kernels do not always exploit well
- FP8 and related low-precision formats need careful layout, scale handling, and launch strategy
- DeepGEMM tries to push that path harder than a more general backend

So the intended advantages are:
- better throughput on supported GPUs
- very strong performance on supported FP8 workloads
- special handling for modern scale/layout formats like ue8m0
- Blackwell- and Hopper-oriented optimization

Pros of DeepGEMM

1. High peak performance on the right hardware
DeepGEMM is meant for fast GEMM on modern NVIDIA GPUs, especially Hopper and Blackwell.

2. Strong FP8 focus
It is more specialized toward FP8-style execution than a generic fallback backend.

3. Hardware-aware optimization
It is designed for the newer tensor-core-era behavior rather than being a pure generic kernel layer.

4. Good fit for some JIT-style optimized paths
It can generate or select highly optimized implementations for supported cases.

Cons of DeepGEMM

1. More specialization means more constraints
If your shape, dtype, layout, or scale format does not match what it wants, it may need fallback paths or may be less reliable.

2. More sensitivity to scale format
Your warning is exactly this:
DeepGEMM is enabled but the checkpoint scale format is not ue8m0, and SGLang warns that this may degrade accuracy on Blackwell.
That warning comes from model_config.py.

3. More moving parts
Because it is more specialized, there is more room for hardware-specific behavior to differ from local expectations.

4. Harder to reason about than a simple fallback backend
If you are debugging correctness, a more general backend can be easier to trust as a baseline.

Difference between DeepGEMM and Triton

DeepGEMM:
- specialized library for high-performance GEMM
- strong emphasis on modern NVIDIA GPUs and FP8-style paths
- more constrained by expected tensor format and scale format
- typically chosen when SGLang believes the hardware/path fits well

Triton:
- more general GPU kernel framework
- often used as a fallback because it is flexible and broadly compatible
- easier to use across many shapes and situations
- usually less specialized than DeepGEMM for the newest FP8-heavy hardware paths

In practice:
- Triton is often the safer baseline
- DeepGEMM is often the more aggressive performance path

Difference between DeepGEMM and CUTLASS

CUTLASS:
- NVIDIA’s template-based kernel library
- lower-level and more general than many higher-level dispatch systems
- often very strong for production-grade GEMM implementations
- still hardware-aware, but more library-style and less “special-case FP8 JIT path” than DeepGEMM in spirit

DeepGEMM vs CUTLASS:
- both target high performance
- CUTLASS is more general-purpose kernel infrastructure
- DeepGEMM is more narrowly focused on very fast modern GEMM, especially the FP8-style path SGLang is wiring in here
- DeepGEMM can be faster in the cases it was built for
- CUTLASS can be a more robust general backend in broader cases

Difference between DeepGEMM and other non-DeepGEMM backends

FlashInfer TRTLLM:
- also high-performance and hardware-aware
- in this repo it is described as especially good for Blackwell and low latency in the FP8 GEMM dispatcher at server_args.py
- usually less “mystery JIT path” feeling than DeepGEMM, but still specialized

AITER:
- ROCm-focused, not relevant to your NVIDIA path

Generic fallbacks:
- usually more stable and broadly compatible
- usually less likely to be the absolute fastest on the newest GPUs

Why the warning matters in your case

The most important part for your current investigation is not abstract performance theory. It is this:

In configurer.py, DeepGEMM is globally enabled on Blackwell if available.
In model_config.py, SGLang warns if your checkpoint is not using ue8m0 scale format while DeepGEMM expects it on Blackwell.

That means:
- SGLang itself believes this mismatch can hurt accuracy
- so DeepGEMM is a credible suspect for your accuracy issue
- even if it is not the main GPTQ Marlin path

Practical summary

If you want the shortest useful summary:

DeepGEMM:
- more specialized
- more performance-oriented on Hopper and Blackwell
- especially relevant for FP8 GEMM paths
- can be excellent for speed
- can be more fragile if tensor format does not match what it expects

Triton:
- more general
- more flexible
- often safer as a baseline
- usually not the absolute best specialized path

CUTLASS:
- high-performance NVIDIA kernel library
- strong and general
- often a solid middle ground between flexibility and speed

For your current debugging goal:
- DeepGEMM is not just “another fast kernel”
- it is now one of the top suspects because SGLang explicitly warned that your checkpoint format may be a bad match for it on Blackwell

So if your question is “what is the main difference operationally?” the answer is:

DeepGEMM is the most specialized and potentially fastest of the group for the right FP8-oriented Blackwell/Hopper cases, but it is also the one currently giving you an explicit accuracy-risk warning in your setup.

If you want, the next useful thing I can do is give you a MiniCPM-SALA-specific matrix of:
1. which parts of your current serving path likely use GPTQ Marlin,
2. which parts may still touch DeepGEMM,
3. and which toggles are best for isolating each path cleanly.
