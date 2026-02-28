1.
# Clone repository
git clone -b minicpm_sala https://github.com/OpenBMB/sglang.git
cd sglang

# One-click installation (creates venv and compiles all dependencies)
bash install_minicpm_sala.sh

2. 
for comilation error, run the nvcc command seperately to fix.

3. 
vi /root/sglang/python/sglang/srt/layers/attention/minicpm_backend.py
modify the line 448 to:
if seqlen_q_sparse_tensor.numel() == 0:
    metadata.max_seqlen_q_adjusted = 0
else:
    metadata.max_seqlen_q_adjusted = seqlen_q_sparse_tensor.max().item() * self.heads_per_group

4.
source /root/sglang/sglang_minicpm_sala_env/bin/activate

export MODEL_PATH=/root/models/openbmb/MiniCPM-SALA/

python3 -m sglang.launch_server \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 16384

