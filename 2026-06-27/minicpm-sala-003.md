I need to use the Marlin-repacked weights. Let me check how the actual inference path repacks weights:

Find Marlin repack code

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization)

Searched for regex `marlin_repack|gptq_marlin|repack.*weight` (`**/python/sglang/srt/layers/quantization/**`), 20 results

Searched for regex `process_weights_after_loading|process_weights` (`**/python/sglang/srt/layers/quantization/**`), 20 results

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#65-65), lines 65 to 100

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#522-522), lines 522 to 600

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils_fp8.py#1-1), lines 1 to 70

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/gptq.py#694-694), lines 694 to 800

Searched for files matching `**/sgl-kernel/**/gptq_marlin_repack*`, 1 match

Searched for text `gptq_marlin_repack` (`**/sgl-kernel/**/*.py`), 6 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin_repack.cu#1-1), lines 1 to 150

Read [](file:///home/oldzhu/sglang/sgl-kernel/python/sgl_kernel/marlin.py)

Read [](file:///home/oldzhu/sglang/sgl-kernel/tests/test_marlin_repack.py#91-91), lines 91 to 149

Searched for text `gptq_marlin_repack` (`**/sgl-kernel/**/*.cc`), 3 results

Searched for text `gptq_marlin_repack` (`**/sgl-kernel/csrc/gemm/marlin/**/*.cc`), no results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/common_extension.cc#200-200), lines 200 to 250

Searched for regex `^(at::Tensor|torch::Tensor).*gptq_marlin_repack` (`**/sgl-kernel/csrc/gemm/marlin/**/*.cu`), no results

Searched for text `gptq_marlin_repack` (`**/sgl-kernel/csrc/**/*.h`), no results

Searched for text `gptq_marlin_repack` (`**/sgl-kernel/csrc/**`), 8 results

Read [](file:///home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin_repack.cu#240-240), lines 240 to 330

Now I understand. Let me update the benchmark to use `gptq_marlin_repack` for proper Marlin comparison:

Read [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py)

Now I need to add Marlin repacking and benchmarking. Let me rewrite the benchmark to do both:

Created [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py)

Ran terminal command:  rm /home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py

Created [](file:///home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py)

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, upload_file
base_url, token = load_config()
upload_file(base_url, token,
    '/home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py',
    '/root/data/bench_w4a8_vs_marlin_real_weights.py')
print('upload done')
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 bench_w4a8_vs_marlin_real_weights.py 2>&1'

The ScalarType needs `has_bias=False` for GPTQ (zero-point based). Let me check:

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sglang.srt.layers.quantization.gptq import ScalarType; st = ScalarType.int_(4, False); print(st, st.id, st.has_bias)"')
print(out)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sglang.srt.layers.quantization.gptq import ScalarType; st = ScalarType.int_(4, False); print(st, st.id, st.has_bias())"')
print(out)
"






```

```


Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, upload_file
base_url, token = load_config()
upload_file(base_url, token,
    '/home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py',
    '/root/data/bench_w4a8_vs_marlin_real_weights.py')
print('upload done')
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 bench_w4a8_vs_marlin_real_weights.py 2>&1'

It needs `is_zp_float=False` explicitly. Let me check how the actual code calls it:

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#464-464), lines 464 to 510

I see — the `gptq_marlin_gemm` function takes `is_zp_float` but `apply_gptq_marlin_linear` doesn't expose it. The issue is that the `wtype` check happens inside `gptq_marlin_gemm`. Let me look at what the wtype check expects:

Read [](file:///home/oldzhu/sglang/python/sglang/srt/layers/quantization/marlin_utils.py#510-510), lines 510 to 540

The `forward_context` is likely None in our standalone script. Let me check what `get_forward_context()` returns:

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sglang.srt.compilation.piecewise_context_manager import get_forward_context; print(get_forward_context())"')
print(out)
"

Good, so it goes through `gptq_marlin_gemm` directly. Let me check the `gptq_marlin_gemm` signature:

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel import gptq_marlin_gemm; import inspect; print(inspect.signature(gptq_marlin_gemm))"')
print(out)
"

I see — `b_q_type` is `sgl_kernel.scalar_type.ScalarType`, not the one from gptq.py. Let me check:

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel import ScalarType; st = ScalarType.int4; print(st, type(st))"')
print(out)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel import gptq_marlin_gemm; from sgl_kernel.scalar_type import ScalarType; st = ScalarType.int4; print(st, type(st))"')
print(out)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel.scalar_type import ScalarType; print([x for x in dir(ScalarType) if not x.startswith(\\"_\\")])"')
print(out)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel.scalar_type import ScalarType; print([a for a in dir(ScalarType) if not a.startswith(\\"_\\")])"')
print(out)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel.scalar_type import ScalarType; print([a for a in dir(ScalarType)])"')
print(out)
"

Same structure. Let me just call `gptq_marlin_gemm` directly with the right ScalarType:




```

```


Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, upload_file
base_url, token = load_config()
upload_file(base_url, token,
    '/home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py',
    '/root/data/bench_w4a8_vs_marlin_real_weights.py')
print('upload done')
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 bench_w4a8_vs_marlin_real_weights.py 2>&1'

The error says it must be `u4` not `int4`. Let me check the ScalarType values:

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel.scalar_type import ScalarType; print(ScalarType.int_(4, False)); st = ScalarType.int_(4, False); print(\\"id:\\", st.id, \\"size_bits:\\", st.size_bits, \\"has_bias:\\", st.has_bias())"')
print(out)
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel.scalar_type import ScalarType; st = ScalarType.int_(4, False); print(st, st.id, st.size_bits, st.has_bias())"')
print(out)
"

The issue is that the Marlin C++ kernel expects `u4` (unsigned) not `int4` (signed). Let me check what `uint` gives:

Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, exec_command
base_url, token = load_config()
code, out = exec_command(base_url, token,
    'python3 -c "from sgl_kernel.scalar_type import ScalarType; print(ScalarType.uint(4, False))"')
print(out)
"




```

```


Ran terminal command:  cd /home/oldzhu/sglang && python3 -c "
from scripts.fcloud.fcloud_exec import load_config, upload_file
base_url, token = load_config()
upload_file(base_url, token,
    '/home/oldzhu/sglang/benchmark/soar/demo_sala/bench_w4a8_vs_marlin_real_weights.py',
    '/root/data/bench_w4a8_vs_marlin_real_weights.py')
print('upload done')
"

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 bench_w4a8_vs_marlin_real_weights.py 2>&1'

Now let me run the same comparison for the other shapes:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 -c "
import sys
sys.argv = ["", "/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8", "0", "self_attn.o_proj"]
exec(open("bench_w4a8_vs_marlin_real_weights.py").read())
" 2>&1'

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 -c "
import sys
sys.argv = ["", "/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8", "0", "self_attn.q_proj"]
exec(open("bench_w4a8_vs_marlin_real_weights.py").read())
" 2>&1'

The q_proj has a different g_idx layout (not full K). Let me skip that and run down_proj:

Ran terminal command:  cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec 'cd /root/data && SOAR_W4A8_FUSED_SO=/root/submission_sim/libw4a8_fused_gemm.so python3 -c "
import sys
sys.argv = ["", "/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8", "0", "mlp.down_proj"]
exec(open("bench_w4a8_vs_marlin_real_weights.py").read())
" 2>&1'

Now I have the complete picture. Here's the definitive comparison:

## Fused W4A8 REAL vs Marlin — Large-M Throughput (TFLOPS)

### o_proj (N=4096, K=4096)
| M | Fused | Marlin | Speedup (Marlin) |
|---|-------|--------|-----------------|
| 4096 | 40.4 | **179.0** | **4.4x** |
| 8192 | 41.3 | **158.9** | **3.8x** |
| 16384 | 45.4 | **135.5** | **3.0x** |
| 65536 | 37.0 | **135.7** | **3.7x** |

### down_proj (N=4096, K=16384)
| M | Fused | Marlin | Speedup (Marlin) |
|---|-------|--------|-----------------|
| 4096 | 34.9 | **135.4** | **3.9x** |
| 8192 | 36.1 | **81.6** | **2.3x** |
| 16384 | 39.1 | **82.3** | **2.1x** |
| 65536 | 37.4 | **82.7** | **2.2x** |

### gate_up (N=32768, K=4096)
| M | Fused | Marlin | Speedup (Marlin) |
|---|-------|--------|-----------------|
| 4096 | 46.1 | **136.1** | **3.0x** |
| 8192 | 49.2 | **136.3** | **2.8x** |
| 16384 | 50.1 | **136.3** | **2.7x** |
| 65536 | 38.3 | **136.3** | **3.6x** |

**Conclusion**: The fused W4A8 REAL kernel is **2-4x slower** than Marlin at every measured shape and M value. Despite using FP8 tensor cores (296 TFLOPS peak), it only achieves 35-50 TFLOPS (12-17% utilization), while Marlin achieves **80-179 TFLOPS** on the same SM120 hardware.

The fused kernel is **not competitive** with Marlin in either speed or accuracy. This strongly suggests we should **abandon the fused W4A8 REAL path** and focus optimization efforts elsewhere.

Shall I proceed with updating the docs and pausing the instance?

