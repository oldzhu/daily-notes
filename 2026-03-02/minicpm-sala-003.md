source /root/sglang/sglang_minicpm_sala_env/bin/activate
export MODEL_PATH=/root/models/openbmb/MiniCPM-SALA/
python3 -m sglang.launch_server  \ 
--model-path /root/models/openbmb/MiniCPM-SALA \   
--host 0.0.0.0   \
--port 30000   \
--trust-remote-code \  
--disable-radix-cache \  
--attention-backend minicpm_flashinfer \  
--chunked-prefill-size 8192 \
--skip-server-warmup \
--dense-as-sparse

download eval data and eval script and bench script
download https://github.com/OpenBMB/SOAR-Toolkit/raw/refs/heads/main/demos/demo-quant.tar.gz
download https://github.com/OpenBMB/SOAR-Toolkit/raw/refs/heads/main/demos/demo-sala.tar.gz
extract to demo-quant and demo-sala
export MODEL_DIR=/root/models/openbmb/MiniCPM-SALA/
export DATA_DIR=/root/data
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path ${MODEL_DIR} \
  --data_path ${DATA_DIR}/perf_public_set.jsonl \
  --concurrency 32
...
2026-03-02 08:38:01,475 - SGLANG_INFERENCE - INFO - SGLang sampling kwargs: {'temperature': 0.0, 'max_tokens': 65536, 'stop': ['<|im_end|>', '</s>']}
  Sending 150 requests to SGLang API (concurrency=32)...
Generating:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 138/150 [35:00<08:13, 41.13s/it]


Generating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [55:44<00:00, 22.30s/it]
2026-03-02 09:33:46,265 - SGLANG_INFERENCE - INFO - Processing time: 3344.79s

Generation completed in 3350.28 seconds

--- Evaluation Results ---

Average Score: 80.67%
Total Duration: 3350.28 s
Total Tokens: In=8644166, Out=1039014
Average Tokens/Sample: In=57627.8, Out=6926.8
Overall TPS (Output): 310.13 tokens/s
Detailed results saved to outputs/20260302_083754/predictions.jsonl
...
cat > /root/test_data.jsonl << 'EOF'
{"question": "Hello, how are you?", "model_response": "I am doing well, thank you for asking!"}
{"question": "What is the capital of France?", "model_response": "The capital of France is Paris."}
{"question": "Explain the concept of machine learning in simple terms.", "model_response": "Machine learning is a type of artificial intelligence where computers learn from data to make decisions or predictions, without being explicitly programmed for each specific task. It's like teaching a computer to recognize patterns and improve its performance through experience."}
{"question": "What are the three states of matter?", "model_response": "The three states of matter are solid, liquid, and gas. Solids have a fixed shape and volume, liquids have a fixed volume but take the shape of their container, and gases have neither fixed shape nor volume."}
{"question": "Who wrote the novel 'Pride and Prejudice'?", "model_response": "The novel 'Pride and Prejudice' was written by Jane Austen. It was first published in 1813 and is considered one of the most beloved works of English literature."}
{"question": "中国的首都是哪里？", "model_response": "中国的首都是北京。北京位于华北平原北部，是中国政治、文化和国际交流中心。"}
{"question": "请解释什么是人工智能？", "model_response": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统，如学习、推理、问题解决、理解自然语言和识别图像等。"}
{"question": "《红楼梦》的作者是谁？", "model_response": "《红楼梦》的作者是曹雪芹。这部小说是中国古典四大名著之一，被誉为中国古典小说的巅峰之作。"}
{"question": "水的化学式是什么？", "model_response": "水的化学式是H2O，由两个氢原子和一个氧原子组成。水是地球上最常见的物质之一，对所有生命体都至关重要。"}
{"question": "请列举三种常见的编程语言。", "model_response": "三种常见的编程语言是：Python、Java和JavaScript。Python以简洁易学著称，Java广泛用于企业级开发，JavaScript则是Web开发的核心语言。"}
EOF

# Run benchmark
export SPEED_DATA_S1=/root/data/test_data.jsonl
export SPEED_DATA_S8=/root/data/test_data.jsonl
export SPEED_DATA_SMAX=/root/data/test_data.jsonl

/root/data/bench_serving.sh http://127.0.0.1:30000
...
============================================================
  Benchmark Duration 汇总
============================================================
    S1:       5.57s
    S8:       1.46s
  Smax:       1.07s
============================================================
{"S1": 5.57, "S8": 1.46, "Smax": 1.07}
...
