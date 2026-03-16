I’m doing a read-only pass over the current fp8 + gpu graph + force-dense-minicpm path, the official guidance points you quoted, and your hardware notes so I can recommend the highest-ROI next optimization before any code changes.

Reviewed 3 files and searched for regex patterns
Compacted conversation

Explored hardware specifics and gathered targeted information
I’ve confirmed the current question is mostly a prioritization problem: squeeze more speed from the stable dense path now, or pivot to a riskier sparse/speculative path. Next I’m reading the hardware note, the MiniCPM Lightning path, and the current Marlin/FlashInfer tuning hooks to see where there is real headroom.

Optimizing tool selection...

Searched for regex and reviewed minicpm.py lines 180-520
Fetched 2 resources
Reviewed rtx-pro-sm120-6000d.md
The official pages still point to the same two recommended routes, but your current score profile changes the decision: with correctness already near the floor, the next step should favor low-risk runtime wins before a large algorithmic pivot. I’m checking whether prefill/decode still has untuned knobs on the current dense path, especially Lightning state updates, FlashInfer split tiling, piecewise CUDA graph, and Marlin kernel selection.

Reviewed 4 files
The current submission script is still a conservative profile: dense MiniCPM, FP8 KV, no server warmup, and a single-request prefill cap. I’m now checking whether the remaining speed headroom is mostly in dense prefill scheduling or in Lightning decode/state-update kernels.

Searched for multiple regex patterns and reviewed a file
I’ve narrowed likely headroom to three areas: dense prefill capture, FlashInfer tile sizing, and Lightning decode state kernels. I’m reading the actual fast path implementations now so the recommendation is based on the code that runs, not just flags.

Reviewed 4 files
The Marlin side is still using a small, static set of thread/block configurations inherited from older architectures, while the Lightning path is doing per-batch state loads and stores on every layer. I’m checking whether those are exposed enough to be a realistic week-3 optimization, or whether they’re too invasive compared with a safer runtime-tuning pass.

Finished with 15 steps
I have enough to rank the options. I’m doing one last targeted lookup for exact code anchors so the recommendation can point to the real bottlenecks and the safest next iteration to implement if you approve.

Finished with 9 steps
One important constraint surfaced in code: piecewise CUDA graph is not a free switch here. On the current MiniCPM hybrid path it is likely disabled by layer-shape checks unless we add explicit support for Lightning layers, so the easiest remaining gains are probably scheduler and warmup tuning first, not prefill graph capture.

Finished with 5 steps
I’ve confirmed two concrete things from code: piecewise CUDA graph is currently blocked by the hybrid Lightning layers, and server warmup is being skipped entirely. I’m checking what that warmup actually covers, because if it includes graph/kernel preparation then skipping it is one of the cleanest possible speed regressions to reverse.

Finished with 10 steps
Your rank improved, but the raw signal is mixed: all three official S scores moved down versus the first official run, and acc_ori also dropped from 80.67 to 80.07. That means the move from 10th to 6th is good for standings, but it is not evidence that the current dense MiniCPM profile is already the fastest path. With the deadline this close, I would not pivot to sparse or EAGLE3 yet.

The current best next move is still on the dense path, but not by jumping straight into Lightning or speculative work. The most likely low-risk headroom is in serving behavior, not model math. Your current submission profile in prepare_env.sh:41 is very conservative for throughput: it forces prefill_max_requests to 1, caps max_running_requests at 20, and skips server warmup entirely. In code, prefill_max_requests really does hard-stop batching in the scheduler at schedule_policy.py:580, and schedule_conservativeness directly scales the token reservation ratio at scheduler.py:817. Also, skipping warmup bypasses the startup request at http_server.py:1470 and http_server.py:1627, which is one of the easiest ways to pay first-request graph and kernel setup cost during evaluation.

On the code-path question, there is still room in Lightning and Marlin, but they are second-step optimizations, not first-step ones. The Lightning path in hybrid_linear_attn_backend.py:1577 loads recurrent state per layer, chooses fused_recurrent versus chunk with a hard threshold at hybrid_linear_attn_backend.py:1589, then stores final state back at hybrid_linear_attn_backend.py:1619. That is a real hotspot candidate, but it needs profiling and is easy to destabilize. Marlin also has room: the kernel chooser in gptq_marlin.cu:141, gptq_marlin.cu:149, and gptq_marlin.cu:473 still uses a tiny static config set inherited from older GPUs, so an SM120-specific tile and warp search is plausible. But that is CUDA-kernel work and not what I would burn the next iteration on before trying the easier serving-side gains.

I would not move to sparse now. You still have the FP8 plus GPU graph plus sparse correctness issue unresolved, and the sparse path has known extra routing overhead from earlier investigation. I also would not start 路径二 now. The official guidance is right that Lightning-compatible EAGLE3 needs algorithm work, and the verify path complexity is real. The current hybrid stack already has special target_verify handling, but this is not a one-day optimization. With less than two days left, it is too risky.

One important constraint: piecewise CUDA graph is attractive in theory for long-prefill workloads, but it is not a drop-in win on this model. Standard CUDA graph only captures decode at model_runner.py:1967. Piecewise support exists in server_args.py:941 and server_args.py:1032, but MiniCPM hybrid layers currently fail the standard-GQA check in model_runner.py:2020, so this is not the fast next lever unless we explicitly add hybrid-layer support.

Proposal for the next single optimization feature, no edits yet:

Objective and expected gain
Dense-path scheduler and warmup tuning for the existing fp8 plus gpu graph plus force-dense-minicpm profile. Expected gain is modest but realistic: mainly S8 and Smax improvement from better prefill concurrency and request packing, with possible S1 improvement if warmup removes first-run costs. Accuracy risk should be near zero because this does not change model math.
Rule-compliance check
This stays within official runtime optimization scope, keeps fixed evaluation concurrency untouched, does not re-enable prefix cache, does not change the base model, and is reproducible through prepare_env.sh only.
Risk to accuracy and stability
Accuracy risk is low. Main risks are OOM, more fragmentation, or worse tail latency if prefill concurrency is set too aggressively.
Exact files to change
prepare_env.sh
English and Chinese docs for a new CHANGE_0042 pair under docs/soar_2026_changes.
Exact tuning plan
Remove skip-server-warmup.
Sweep prefill_max_requests from 1 to 2 to 4.
Sweep max_running_requests from 20 to 28 to 36.
Sweep schedule_conservativeness from 1.0 down to 0.85 and 0.7.
Keep force-dense-minicpm and fp8_e5m2 unchanged.
Leave sparse and speculative off.
Validation commands
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_DIR> --data_path <DATA_DIR>/perf_public_set.jsonl --concurrency 32
export SPEED_DATA_S1=<path>
export SPEED_DATA_S8=<path>
export SPEED_DATA_SMAX=<path>
bash SOAR/bench_serving.sh http://127.0.0.1:30000
7. Success and failure criteria
Success means equal or better public accuracy and better S8 or Smax without collapsing S1.
Failure means no speed gain, OOM, unstable warmup, or accuracy drift.

Answering your direct question: yes, there is still something to do for both Lightning Attention and SM120 Marlin tuning, but not first. If this serving-side tuning gives only small gains, I would rank the next deeper branches as:

SM120-aware Marlin tile and warp tuning
Lightning SimpleGLA state-path optimization
Sparse FP8 plus GPU graph recovery
EAGLE3
If you want, I can now prepare the formal CHANGE_0042 implementation proposal around dense-path scheduler and warmup tuning and wait for your approval before editing.


