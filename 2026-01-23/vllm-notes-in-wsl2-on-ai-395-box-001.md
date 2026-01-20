A/B/C are three “entry paths” into vLLM depending on what kind of work you want to own.

A) OpenAI server / streaming (“frontend & serving”)

What it is: everything from HTTP request → engine request → streaming tokens back (SSE/websocket-ish patterns, auth, request schemas, compatibility).
Where it lives: openai, plus some shared entrypoint utilities in entrypoints.
Typical tasks: add/adjust OpenAI API behaviors, fix streaming edge cases, improve error messages/validation, add metrics/logging, handle batching/timeout/cancellation correctly.
B) Scheduler / core throughput (“engine core”)

What it is: the heart of vLLM performance/correctness—continuous batching, token scheduling, preemption, KV cache budgeting, step loop behavior.
Where it lives: core loop in core.py, scheduling in sched, request state in request.py, KV cache in kv_cache_manager.py, executors in executor.
Typical tasks: tune scheduling policy, fix fairness/latency regressions, add instrumentation, improve preemption behavior, correctness fixes that require understanding request lifecycle.
C) Attention / kernels (“low-level performance”)

What it is: the compute-heavy parts—attention backends, fused kernels, custom ops (CUDA/ROCm/CPU), and their integration into Python layers.
Where it lives: attention selection/backends in attention, Python↔native glue in _custom_ops.py, native code in csrc, plus optimized layers/quantization under model_executor and compilation/CUDA graphs under compilation.
Typical tasks: implement/optimize kernels, add a new attention backend, fix kernel correctness issues, improve CUDA graphs/compile compatibility.
How to choose (quick guidance)

Start with A if you care about product/API behavior and can iterate without GPUs sometimes (lots of unit tests, schema work).
Start with B if you want to own “how vLLM schedules tokens” (high impact, more conceptual, needs reading/debugging).
Start with C if you want to do performance engineering and are comfortable with CUDA/ROCm/Triton and profiling.
If you tell me your goal (serving features vs throughput vs kernel perf) and whether you have a GPU locally, I’ll propose 2–3 concrete “starter tasks” and exactly which tests to run for each path.


