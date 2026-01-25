Extended the backend seam to nn::bmm (CPU-only) and pushed it.

nn::bmm now calls backend::get().matmul2d_fwd/bwd once per batch (so CUDA/HIP can later accelerate via the same seam) in ops.cpp.
Added a new counting-backend test test_backend_dispatch_bmm() in test_main.cpp that asserts:
forward calls matmul2d_fwd exactly B times
backward calls matmul2d_bwd exactly B times
Built + ran ctest -C Debug successfully.
Pushed commit: 5adcee6 — “core: route bmm through backend seam”.
