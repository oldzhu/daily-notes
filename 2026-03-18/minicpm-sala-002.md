Yes, there is extra cost, but in the form I’m proposing it should be very small, and usually worth paying if it is done correctly.

The main point is this: a single static config is almost never optimal across all GEMM shapes in your serving workload. Your shapes vary at least in:
- M, which changes with batch and split behavior
- sometimes effective shape mix across prefill, decode, and graph-captured batches
- shared-memory pressure and occupancy behavior for different tile choices

So your intuition is right: there usually is no one static Marlin config that is best for all shapes.

Whether the scoring cost is worth it depends on where and how often you pay it:

1. If you score on every single GEMM launch with no cache, that is a bad design.
2. If you score once per distinct shape/config key and cache the result, the overhead is usually negligible.
3. If most hot shapes repeat, the cost becomes close to one-time warmup cost, not steady-state serving cost.

That is why the right design is not “dynamic search every launch,” but:
- build a small candidate set
- score once for a shape key like M/N/K plus relevant flags
- cache the chosen config
- reuse it for later launches of the same shape

In that form, the overhead is typically tiny compared with the GEMM itself. The scoring logic is just a few integer calculations and maybe one occupancy query over a small number of candidates. That is orders of magnitude cheaper than the actual quantized GEMM.

So the tradeoff is:

- Static config:
  - zero selection overhead
  - simple
  - often leaves performance on the table for many shapes

- Auto-scored cached config:
  - small one-time overhead per unique shape
  - better chance of matching the right kernel to the actual workload
  - much safer than hardcoding one “best” SM120 config and hoping it generalizes

The real risk is not the scoring cost itself. The real risk is that even a better Marlin choice may still not move end-to-end S1/S8/Smax much if Marlin is no longer the dominant bottleneck in your current profile.

So my practical answer is:
- yes, there is a small extra cost
- no, that cost is usually not the reason to avoid it
- the important question is not “is scoring too expensive?”
- the important question is “does better Marlin selection move enough real workload time to matter?”

If we do this, I would strongly recommend a cached selector. Then the scoring cost is small enough that it is not the main concern. The main concern becomes whether the measured speedup is large enough to justify another iteration.
