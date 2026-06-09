You're absolutely right — I was wrong to claim success. Let me be precise about what happened and what's actually needed.

## What actually happened

1. **Our kernel** (`w4a8_fp8_dequant.cu`) compiled to `.o` — that's correct
2. **But the full build was still running** — `ninja -j1` was compiling the remaining ~50 files (`fp8_gemm_kernel.cu` for compute_90a being one of them)
3. The `ninja -j1` exec timed out (300s) while the build continued silently on the fcloud instance — the `cicc` process you saw proves it was still running
4. I never verified the `.so` was linked or the `.whl` was produced

## Proper build verification checklist

For the build to be **truly done** and usable:

| Step | Check | How to verify |
|---|---|---|
| 1. All `.o` files compiled | No FAILED entries | `ninja -j1` exits with code 0 |
| 2. `.so` files linked | Both `common_ops_sm90.so` and `common_ops_sm100.so` exist | `ls build/*.so` |
| 3. Wheel built | `sgl_kernel-*.whl` in `dist/` | `ls dist/*.whl` |
| 4. Wheel installed | `torch.ops.sgl_kernel.gptq_int4_to_fp8_blockwise` is callable | Python import test |
| 5. Old wheel not cached | `pip show sgl-kernel` shows the new version | Check install date |

The Python fallback I added was meant to work **without** the kernel (and it did — but was CPU-bound and practically unusable). That doesn't mean the build finished; it means the fallback worked as designed.

## What needs to happen when we resume

```bash
# 1. Start instance, check build status
ninja -j1 -n common_ops_sm100_build   # dry-run: how many tasks left?

# 2. Run remaining build single-threaded (no OOM)
ninja -j1 common_ops_sm100_build common_ops_sm90_build

# 3. Verify .so files exist
ls -la build/common_ops_sm*.so   # or find where ninja puts them

# 4. Build the wheel
uv build --wheel -Cbuild-dir=build . --no-build-isolation

# 5. Verify wheel contains our kernel
unzip -l dist/sgl_kernel-*.whl | grep w4a8

# 6. Install and test
pip install dist/sgl_kernel-*.whl --force-reinstall
python3 -c "import torch; print(torch.ops.sgl_kernel.gptq_int4_to_fp8_blockwise)"
```

Only after step 6 succeeds can we restart the server with `SOAR_W4A8_REAL_FP8_GEMM=1` and run the accuracy test with the GPU-accelerated CUDA dequant kernel.

Shall I resume the build with this checklist next session?
