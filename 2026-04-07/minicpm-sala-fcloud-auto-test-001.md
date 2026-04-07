while we are waiting for the testing result, let us working on another feature - fcloud utility to auto the testing before continue to improve benchmark and keep accuracy >= 80%. At the current, every new testing need man in the loop, u commit and push the new change, I pull the change from fcloud and then test and then feedback the result to you, you continue the improving and change, and then commit and push, it is not efficient. Let us develop a skill or tool script which u can used to run command in fcloud and get the output u want directly, using this tool you can automize the working flow changing-testing-get the result-chaning-testing and only pause to wait I review and approve when necessary.  the background detail of the manual steps is as the below:
1. fcloud is jupiter lab - which have a url and token to access, I will new several terminals concurrently depends on how many commands I want to run concurrently(usually two - one is starting sglang and another is running accuracy evaluation and benchmark testing)
2. in the fcloud instance, the cloned source is at the folder /root/sglang-minicpm and I will run git pull to get the change u made(git will popup a window to ask to modify pull msg and save - I will save directly)
3. the working folder is at /root/submission_sim, I will copy the modified files to the  /root/submission_sim to update the previous file.
for examples: 
cp /root/sglang-minicpm/benchmark/soar/demo_sale/prepare_env.sh /root/submission_sim
cp /root/sglang-minicpm/python/sglang/srt/models/minicpm.py /root/submission_sim/sglang/python/sglang/srt/models/minicpm.py
for the modified files under /root/sglang-minicpm/benchmark/soar/demo_sale/, we will copy to /root/submission_sim
for the modified files under  /root/sglang-minicpm/python, we will copy to /root/submission_sim/sglang/python corresponding place.
in case we modified files under sgl-kernel, after we pull the changes in fcloud, we need to build sgl-kernel wheel from source under /root/sglang-minicpm/sgl-kernel, and then copy the build whl from /root/sglang-minicpm/sgl-kernel/dist to /root/submission_sim for testing.
4. after update the files under /root/submission_sim, normally we will run the below commands to start sglang:

cd /root/submission_sim
source ./prepare_env.sh
MODEL_PATH=/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8
HOST=0.0.0.0
PORT=30000
pkill -f "sglang.launch_server" || true
read -r -a EXTRA_ARGS <<< "${SGLANG_SERVER_ARGS:-}"
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"

5. in another terminal, run the below command to do accuracy and benchmark testing
cd /root/data

//get the accuracy json output
python3 eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
  --data_path /root/submission_sim/perf_public_set.jsonl \
  --concurrency 32

//get the s1 duration json output
python3 run_soar_suite.py --api-base http://127.0.0.1:30000 \
--model-path /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
--speed-data-s1 /root/data/benchmark/soar/data/speed_s1.jsonl

//get the s8 duration json output
python3 run_soar_suite.py --api-base http://127.0.0.1:30000 \
--model-path /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
--speed-data-s8 /root/data/benchmark/soar/data/speed_s1.jsonl

//get the smax duration json output
python3 run_soar_suite.py --api-base http://127.0.0.1:30000 \
--model-path /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
--speed-data-smax /root/data/benchmark/soar/data/speed_s1.jsonl

If need to modify the commands or add new commands to test, will let you know. 
