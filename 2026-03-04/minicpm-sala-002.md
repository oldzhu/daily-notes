1. 
export MEM_FRACTION_STATIC=0.7
./launch_toolkit-default.sh

2. 
root@ai-e7e98a7c52:~/data# python3 run_soar_suite.py --api-base http://127.0.0.1:30000 --model-path /root/models/openbmb/MiniCPM-SALA --speed-data-s1 /root/data/benchmark/soar/data/speed_s1.jsonl --num-prompts 8

3. 
#!/bin/bash
#
python3 generate_speed_datasets.py --profile quick10 --rows-s1 24 --rows-s8 32 --rows-smax 40
run_gen_data.sh

4.
# another default minimum launch without oom --max-running-requests 20   --mem-fraction-static 0.84   --schedule-conservativeness 1.0 
# it is wired original default could start at morning but fail with oom afternoon. 
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --max-running-requests 20 \
  --schedule-conservativeness 1.0 \
  --skip-server-warmup \
  "${extra_args[@]}"
actually there is no --mem-fraction-static 0.84 set it also starts just fine. I am testing a new eval.

5. 
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 83.05 GiB of which 140.38 MiB is free. Including non-PyTorch memory, this process has 23.33 GiB memory in use. Of the allocated memory 21.92 GiB is allocated by PyTorch, with 104.00 MiB allocated in private pools (e.g., CUDA Graphs), and 797.79 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
export  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
