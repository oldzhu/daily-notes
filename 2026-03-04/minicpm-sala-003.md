Your observation is valid, and not weird for long-context workloads.

Why morning worked, afternoon failed (same-ish config)

Dynamic memory pressure differs run-to-run: request mix/lengths and concurrency timing change peak activation memory.
Allocator fragmentation accumulates over long runs/restarts, especially with variable sequence lengths.
Auto heuristics are sensitive: auto-chosen memory pool/chunk settings can be near cliff edge; small runtime variance flips from pass to OOM.
CUDA graph/private pools and reserved-but-unallocated segments can shift effective free contiguous blocks.
Why your new options helped startup OOM

--max-running-requests 20 reduces scheduler in-flight pressure, so less simultaneous prefill/decode memory demand.
--schedule-conservativeness 1.0 tends to avoid aggressive batching bursts, smoothing peaks.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True mitigates fragmentation by allowing allocator segments to grow more flexibly, reducing “can’t find contiguous chunk” failures.
Why eval can still OOM later

Startup success only proves static pool init is okay.
During eval, a pathological batch (long prompt(s) + concurrent requests) can still hit transient peak allocation (your 768 MiB failure).
Practical stable recipe (for baseline collection)

Keep:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
--attention-backend minicpm_flashinfer
--max-running-requests 20 (or 16 if unstable)
--schedule-conservativeness 1.0~1.2
Add if needed:
--prefill-max-requests 1
--chunked-prefill-size 4096 (or 2048 for max safety)
Then do eval ramp 8 -> 16 -> 32 before full bench.
This is exactly the type of “near-boundary memory behavior” we should capture in the next optimization notes once you send the new run logs.
