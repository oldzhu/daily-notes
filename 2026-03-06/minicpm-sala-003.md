1. vi test_gptq_001.sh
#!/bin/bash
#
export SOAR_QUANT_MODE=gptq
export SOAR_GPTQ_CALIBRATION_FILE=/root/data/perf_public_set.jsonl
export SOAR_GPTQ_CALIBRATION_FIELD=question
export SOAR_GPTQ_CALIBRATION_SAMPLES=128
export SOAR_GPTQ_BITS=4
export SOAR_GPTQ_GROUP_SIZE=128
export SOAR_GPTQ_BATCH_SIZE=2

bash /root/sglang/benchmark/soar/demo_sala/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/openbmb/MiniCPM-SALA-gptq

2. 
source /root/sglang/sglang_minicpm_sala_env/bin/activate
uv pip install gptqmodel --no-build-isolation -v
...
  /root/.cache/uv/sdists-v9/pypi/gptqmodel/5.7.0/8bsqPmpfke5PLcLh3X1Ob/src/gptqmodel_ext/marlin/kernel_bf16_ku8b128.cu(85): error: function
      "marlin::Marlin<scalar_t,w_type_id,s_type_id,threads,thread_m_blocks,thread_n_blocks,thread_k_blocks,m_block_size_8,stages,group_blocks,is_zp_float>(const int4 *, const
      int4 *, int4 *, int4 *, const int4 *, const int4 *, const uint16_t *, const int4 *, const int *, int, int, int, int, int, int *, __nv_bool, __nv_bool, __nv_bool, int) [with
      scalar_t=nv_bfloat16, w_type_id=1125899923621888L, s_type_id=1125899906909960L, threads=128, thread_m_blocks=4, thread_n_blocks=4, thread_k_blocks=8, m_block_size_8=false,
      stages=4, group_blocks=8, is_zp_float=false]" cannot be instantiated -- no template definition was supplied
        template __attribute__((global)) void Marlin<nv_bfloat16, vllm::kU8B128.id(), vllm::kBFloat16.id(), 128, 4, 4, 8, false, pipe_stages, 8, false>( const int4 *__restrict__
      A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp, const int4 *__restrict__ b_bias_ptr, const int4 *__restrict__ scales_ptr, const uint16_t
      *__restrict__ scale2_ptr, const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks, bool
      has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem );
                 ^

      75 errors detected in the compilation of "/root/.cache/uv/sdists-v9/pypi/gptqmodel/5.7.0/8bsqPmpfke5PLcLh3X1Ob/src/gptqmodel_ext/marlin/kernel_bf16_ku8b128.cu".
      ninja: build stopped: subcommand failed.

...
  W0306 09:08:16.388000 4790 torch/utils/cpp_extension.py:531] There are no c++ version bounds defined for CUDA version 12.8
      Traceback (most recent call last):
        File "<string>", line 886, in run
        File "<string>", line 466, in _download_with_progress
        File "/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/urllib/request.py", line 276, in urlretrieve
          raise ContentTooShortError(
      urllib.error.ContentTooShortError: <urlopen error retrieval incomplete: got only 146800640 out of 157925407 bytes>

      During handling of the above exception, another exception occurred:

      Traceback (most recent call last):
        File "/root/sglang/sglang_minicpm_sala_env/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2597, in _run_ninja_build
          subprocess.run(
        File "/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/subprocess.py", line 571, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '222']' returned non-zero exit status 255.
...
