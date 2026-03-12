Proposal
Objective: stop doing blind trial-and-error on MiniCPM high-concurrency decode and switch to evidence-driven optimization for the current CUDA-graph and sparse-attention path.

I re-checked the latest official SOAR pages first. The current rules still support this approach: runtime optimization is allowed, fixed concurrency must be respected, prefix cache must stay effectively off during benchmark, correctness must stay above the threshold, and changes must remain reproducible and stable. The latest toolkit still points to quantization and speculative decoding as the main official technical paths, but nothing forbids profiling-guided runtime work on the existing inference path as long as it stays within those constraints.

Rule-compliance check:

Keep the benchmark method aligned with the official 3-tier setup from the toolkit and competition pages.
Do not change the benchmark concurrency definitions.
Do not re-enable forbidden prefix-cache behavior.
Keep accuracy validation in the loop with the public eval script before treating any speed change as acceptable.
No submission packaging changes are needed for this profiling stage.
Risk to accuracy/stability:

Profiling itself has no model-accuracy risk.
Nsight and perf runs can perturb absolute speed a bit, so they are for diagnosis, not final leaderboard numbers.
The only real risk is drawing conclusions from too-light workloads, which is why I am recommending one heavier shared dataset.
Exact repo entrypoints to use, no code edits yet:

generate_speed_datasets.py
run_soar_suite.py
launch_profile.sh
server_args.py
Likely diagnosis targets after traces:
cuda_graph_runner.py
model_runner.py
minicpm_backend.py
minicpm_fuse_kernel.py
minicpm_sparse_kernels.py
Recommended Preset
Use one shared dataset preset instead of three different files.

Preset name: balanced_shared_v1

Reason:

quick10 is only a smoke test.
balanced is much closer to the official published length distribution and is heavy enough to expose concurrency-path bottlenecks.
Use the generated speed_smax.jsonl as the shared file for all three tiers, because it has the largest row count under the same distribution and stresses the serving path better.
Generate it with:


python3 benchmark/soar/generate_speed_datasets.py \  --profile balanced \  --prompt-source synthetic \  --output-dir benchmark/soar/data/balanced_shared_v1 \  --model-path /root/models/openbmb/MiniCPM-SALA
Then use:


SHARED=benchmark/soar/data/balanced_shared_v1/speed_smax.jsonl
I am not going to invent an “expected runtime on your fcloud box” without one calibration run. That would be fake precision. The defensible way is:

generate the dataset,
read the generator’s printed estimate,
run one baseline suite once,
then use that measured wall time as the planning number for later profiling captures.
Profiling Plan
This is the concrete profiling workflow I recommend for your Nvidia 6000D-class box.

Baseline benchmark plus exported metrics.
Nsight Systems trace during the same workload.
CPU perf run only if the Nsight trace shows GPU idle gaps or host-side stalls.
Optional Nsight Compute only after a single kernel is clearly dominant.
Start the server with the existing probe-style settings plus metrics export. Keep CUDA graph enabled.

Example launch shape:


python3 -m sglang.launch_server \  --model-path /root/models/openbmb/MiniCPM-SALA \  --host 0.0.0.0 \  --port 30000 \  --trust-remote-code \  --disable-radix-cache \  --attention-backend minicpm_flashinfer \  --chunked-prefill-size 32768 \  --max-prefill-tokens 32768 \  --prefill-max-requests 1 \  --max-running-requests 20 \  --mem-fraction-static 0.84 \  --schedule-conservativeness 1.0 \  --skip-server-warmup \  --enable-metrics \  --export-metrics-to-file \  --export-metrics-to-file-dir benchmark/soar/results/request_metrics
That launch shape is aligned with the repo’s probe profile in launch_profile.sh:34.

Run the suite with the one shared file:


SHARED=benchmark/soar/data/balanced_shared_v1/speed_smax.jsonlpython3 benchmark/soar/run_soar_suite.py \  --api-base http://127.0.0.1:30000 \  --model-path /root/models/openbmb/MiniCPM-SALA \  --speed-data-s1 "$SHARED" \  --speed-data-s8 "$SHARED" \  --speed-data-smax "$SHARED" \  --disable-tqdm
Capture coarse GPU telemetry in parallel:


nvidia-smi dmon -s pucvmet -d 1 -o TD -f benchmark/soar/results/nvidia_dmon.log
Nsight Systems Capture
This is the most important step. It tells us whether the bottleneck is host-side, graph-side, or kernel-side.

Launch the server under Nsight Systems in one shell:


nsys profile \  --trace=cuda,nvtx,osrt \  --sample=none \  --cpuctxsw=true \  --cuda-graph-trace=node \  --force-overwrite=true \  --delay 20 \  --duration 90 \  -o benchmark/soar/results/nsys_minicpm_probe \  python3 -m sglang.launch_server \    --model-path /root/models/openbmb/MiniCPM-SALA \    --host 0.0.0.0 \    --port 30000 \    --trust-remote-code \    --disable-radix-cache \    --attention-backend minicpm_flashinfer \    --chunked-prefill-size 32768 \    --max-prefill-tokens 32768 \    --prefill-max-requests 1 \    --max-running-requests 20 \    --mem-fraction-static 0.84 \    --schedule-conservativeness 1.0 \    --skip-server-warmup
In a second shell, start the benchmark after the server is ready and before the capture window expires.

What we want from the trace:

Are there long GPU idle gaps between decode launches?
Are there many tiny kernels with host gaps?
Is CUDA-graph replay visible and dense?
Are there repeated metadata copies or memsets every replay?
Does one kernel dominate decode time?
CPU Profiling
Only do this if Nsight suggests the GPU is waiting on the host.

Find the server PID and sample stacks during the benchmark window:


pidof -s python3perf record -F 99 -g -p <SERVER_PID> -- sleep 60perf report
If you prefer a flamegraph workflow later, that is fine too. This is the closest equivalent to the Windows/Linux CPU-hang workflow you described.

How To Interpret Results
This is where profiling becomes useful instead of guesswork.

If Nsight shows large blank regions on the GPU timeline:

likely host-side bottleneck.
next targets: scheduler, batch assembly, replay metadata setup, Python/C++ dispatch.
likely files:
model_runner.py
cuda_graph_runner.py
minicpm_backend.py
If Nsight shows many short kernels with visible gaps:

likely launch overhead or graph miss/replan overhead.
next targets: graph shape reuse, replay metadata updates, launch batching, per-layer wrapper setup.
If Nsight shows one or two kernels dominate decode time and GPU is busy:

likely kernel bottleneck.
next targets:
minicpm_fuse_kernel.py
minicpm_sparse_kernels.py
plus the corresponding CUDA/Triton kernels behind them.
If Nsight shows frequent memcpy or memset activity around each decode step:

likely metadata serialization / sparse page-table conversion / buffer rewrite overhead.
next targets: replay metadata path, sparse FlashInfer conversion, CPU-to-GPU control traffic.
If CPU perf shows heavy scheduler or Python stacks:

this is a host bottleneck, not a GPU kernel problem.
then we should stop touching attention kernels and work on scheduling or graph-runner overhead first.
Validation Commands
Correctness:


python3 eval_model.py \  --api_base http://127.0.0.1:30000 \  --model_path /root/models/openbmb/MiniCPM-SALA \  --data_path /path/to/perf_public_set.jsonl \  --concurrency 32
Speed:

use the same shared dataset and same launch flags before and after any change.
keep --disable-tqdm only for cleaner logs.
do not use profiling traces to claim final speed; use clean non-profiled reruns for that.
Success And Failure Criteria
Success looks like:

one clear dominant bottleneck class is identified from trace data,
next code target becomes obvious,
and future patches can be justified against measured evidence instead of speculation.
Failure looks like:

using only smoke datasets,
no trace collection,
and continuing to patch kernels or graph code without knowing whether the GPU or CPU is actually limiting throughput.
No code changes are needed for this proposal stage. If you approve this profiling-first step, the next thing I can do is prepare a repo-specific CHANGE document pair proposal for this profiling methodology, or help you interpret the first Nsight/perf outputs once you collect them.


