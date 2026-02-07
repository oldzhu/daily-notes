yesterday we generated the report for the 10 latest commits for “custom ops + native kernels” (GPU + CPU), pls generate the new report today for the commits in “custom ops + native kernels” which after the yesterday's reported commits

Summary

8 custom-ops/native-kernel commits since the previous window (main@4d9513537), led by CPU attention NEON BF16, ROCm AITER import fix, and a refactor removing align block size logic in moe_permute.
Attention-related work dominates the subjects (3 hits), with BF16 and FA3 appearing twice each.
Hot areas: csrc has the largest churn (1178), followed by v1 (173) and _aiter_ops.py (148).
Notable themes

CPU attention improvements: NEON BFMMLA BF16 implementation and s390x BF16 kernel type.
ROCm AITER import regression fix and AITER backend selection.
Attention stack changes: TRTLLM attention disablement with KV transfer, FA3 swizzle revert.
Want me to commit/push this report, or extract a short “what changed” note into a separate summary doc?


