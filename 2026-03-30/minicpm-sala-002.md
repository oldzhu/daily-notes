I’m pulling the exact hot-path locations so I can explain the proposed epilogue change concretely against the current code rather than speaking in abstractions.

Searched for regex `if self.use_output_norm:|if self.use_output_gate:|y, _ = self.o_proj\(o\)|qkv, _ = self.qkv_proj\(hidden_states\)` (`**/python/sglang/srt/models/minicpm.py`), 18 results

Searched for regex `self.z_proj = ColumnParallelLinear|self.o_proj = RowParallelLinear|self.o_norm = RMSNorm` (`**/python/sglang/srt/models/minicpm.py`), 8 results

Yes. I should refine the earlier wording, because there are really two different ideas:

1. a low-risk epilogue fast path inside the existing MiniCPM lightning code
2. a much more aggressive custom fused kernel

What I am proposing next is the first one, not the second.

Right now the hot tail of the lightning layer in minicpm.py to minicpm.py looks like this in effect:

- take the attention output tensor o
- optionally run output RMSNorm
- run z_proj on hidden_states
- compute sigmoid of z
- multiply o by that gate
- run o_proj

The related modules are created earlier in the same class at minicpm.py, minicpm.py, and minicpm.py.

What the next change would look like is not “merge z_proj and o_proj into one GEMM”, because that is not mathematically possible in the current structure:
- z_proj consumes hidden_states
- o_proj consumes the already-computed attention output o

So the realistic low-risk version is:

- keep the exact same math
- keep z_proj and o_proj as separate linear layers
- tighten the post-attention epilogue so it allocates less and moves less data
- make the gate path use a dedicated fast helper instead of the current generic sequence

Concretely, the change would likely be:

- add a helper in minicpm.py for the lightning output epilogue
- keep this order unchanged:
  - o_norm
  - z_proj(hidden_states)
  - sigmoid(z)
  - o *= gate
  - o_proj(o)
- but implement it with:
  - narrower tensor lifetimes
  - in-place gating where safe
  - fewer temporary reshapes / intermediate tensors
  - a dedicated fast path only for the common lightning configuration we are actually serving now

Why this can improve speed:

- This code runs in the lightning path, and under the current force-dense setup that path dominates most decoder layers.
- Even if each layer only saves a small amount of memory traffic, the savings repeat across many layers and every token step.
- The current sequence creates extra temporary tensor work around sigmoid and the gated multiply. That is not as expensive as a GEMM, but it is still repeated constantly.
- Once qk_norm_rope got faster, the next visible overhead tends to move downstream into the rest of the layer epilogue. That is exactly where this change targets.

Why I expect the gain to be modest rather than huge:

- z_proj is still a full linear op.
- o_proj is still a full linear op.
- So we are not removing the main GEMM cost.
- We are mainly reducing glue overhead between them: temporary tensors, memory traffic, and elementwise launch overhead.

So the speed story is:

- not a miracle feature
- probably small-to-moderate gain
- but low risk
- and especially attractive because it sits on a path that is exercised a lot in your current best configuration

The aggressive version would be a real custom fusion, for example a kernel that combines sigmoid gating with the projection epilogue. That could be stronger, but it is a different class of work:
- new kernel or deeper Marlin/Triton integration
- more validation burden
- much higher risk to correctness and stability

Given your current baseline improvement, I think the right next step is still the low-risk epilogue fast path first, because it is the cleanest way to see whether this post-attention tail is worth pursuing further.

So in short, the proposed change is:

- not “fuse two linears”
- instead “optimize the lightning output epilogue around z_proj, sigmoid gating, and o_proj while preserving exact math”

If you want, I can write the next proposal in that stricter form as CHANGE_0061, with the exact helper shape and expected code diff before touching the code.
