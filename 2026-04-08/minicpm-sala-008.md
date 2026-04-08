You're making a very sharp observation. Let me assess each honestly:

**Bug 6 — probably a real bug, but in a different path than Bug 8**

The crash it fixed was in `infllmv2_attn_stage1` (the **prefill** sparse scoring kernel), while Bug 8 is in `compress_k` kernels (the **decode** path). They're different code paths, so Bug 8 can't directly cause the infllmv2 crash.

**However**, your skepticism has merit in a different way: maybe `cu_seqlens_q_adjusted` was actually **correct** for that kernel. Some GQA implementations expect Q reshaped to fold `heads_per_group` into the token dimension — if `infllmv2_attn_stage1` was designed for that layout, then `cu_seqlens_q_adjusted` was right, and Bug 6's "fix" broke it. I'd need to verify whether `q` is reshaped before that call or not. If it's not reshaped, Bug 6 is correct; if it is, Bug 6 introduced a regression.

**Bug 3 — you're right, it didn't help**

The fact that accuracy went from 50.29% → 52.27% (barely noise) after Bug 3 strongly suggests k_scale wasn't the accuracy problem. And for the bf16 test path, k_scale=1.0 so it's a no-op anyway. The fix is harmless but also useless.

**Revised assessment:**

| Fix | Confidence it's correct | Evidence |
|-----|------------------------|----------|
| Bug 8 | Medium | Justified by math, but crash may not have manifested with smaller sequences |
| Bug 6 | **Uncertain** | Could be correct OR could be breaking intended GQA scoring logic |
| Bug 3 | Low relevance | Didn't improve accuracy, no-op for bf16 |
| CHANGE_0070 | **Likely harmful** | Broke captured-graph/replay consistency |
| CHANGE_0071 | Dependent on Bug 8 | Only needed because Bug 8 introduced int64 |

**This reinforces your original plan**: Test 3 with pre-0070 code (which has NONE of these fixes) is the cleanest baseline. If it shows >80%, we know the original sparse code was functional, and we need to be very selective about which fixes to re-apply — ideally re-adding them one at a time with accuracy validation after each.

Ready to start Test 3 on fcloud when you approve.
