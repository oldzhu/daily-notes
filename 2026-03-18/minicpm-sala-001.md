Yes, that is the idea.

“Auto-scored” would mean: instead of returning the first valid candidate in gptq_marlin.cu, the launcher would enumerate all valid configs for the current shape and pick the best one using a score built from:
- M/N tile coverage
- estimated parallelism
- shared-memory pressure / fit
- optionally real occupancy from cudaOccupancyMaxActiveBlocksPerMultiprocessor

That is different from the current behavior in gptq_marlin.cu, where gptq_marlin.cu effectively stops at the first valid config and returns it.

On when it happens: this would be runtime dynamic selection, not compile-time and not a one-time global initialization choice.

More precisely:
- The kernel family itself is fixed at build time by the compiled instantiations in gptq_marlin.cu.
- The choice among those already-compiled variants happens at runtime inside gptq_marlin.cu.
- So for each actual GEMM shape, the launcher can dynamically choose a config based on the current M, N, K, group settings, shared-memory limit, and device.

In practice, for your serving path, that means:
- during startup / warmup / CUDA graph capture, the runtime will choose configs for the shapes encountered there
- during normal serving, if a new shape appears outside the captured graph shapes, it can choose again for that runtime shape
- so it is shape-dependent runtime selection, not a single “initialize once and reuse forever” decision

A compact way to think about it is:
- build time decides what configs exist
- runtime decides which existing config to launch for this specific GEMM

One more important nuance: if we add optional occupancy querying, that query is still launcher-time logic, not kernel initialization logic. It would happen when selecting the candidate for the current shape, just before launch or graph capture for that shape.

So your interpretation is correct.
