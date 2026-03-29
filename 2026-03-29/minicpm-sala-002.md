I’m checking the MiniCPM attention code path itself so I can answer this against the actual implementation here rather than hand-waving about “standard transformer” behavior. Then I’ll map `qk_norm_rope` to the exact operations and explain where compatibility can break.

Searched for regex `q_norm|k_norm|rotary_emb|q_proj|k_proj|v_proj|MiniCPM|Lightning` (`**/python/sglang/srt/models/minicpm*.py`), 200 results

Search complete: Find MiniCPM attention

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#240-240), lines 240 to 380

Searched for files matching `**/minicpm*`, 22 matches

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm3.py#1-1), lines 1 to 300

Read [](file:///home/oldzhu/sglang/python/sglang/srt/models/minicpm.py#1-1), lines 1 to 200

What I meant is: the math idea is standard, but the exact tensor contract is not always standard enough that one model’s fused kernel can be dropped into another model unchanged.

For MiniCPM-SALA in this repo, the active path is the MiniCPM hybrid path in minicpm.py. Inside the Lightning mixer, the sequence is:

1. do one qkv projection
2. split into q, k, v
3. if qk_norm is enabled, apply q_norm and k_norm
4. if rope is enabled, apply RoPE to q and k
5. reshape and send to the attention backend

You can see that directly in minicpm.py.

By contrast, the existing fused qk_norm_rope path already wired in SRT is used by another model, not MiniCPM-SALA, in qwen3_moe.py. So “compatible” means: can MiniCPM-SALA present q and k to that fused kernel in the exact shape, dtype, rotary-dimension policy, and memory layout that kernel expects, without changing semantics.

Why compatibility is not automatic

Even if many transformers conceptually do “q/k norm + RoPE”, there are several ways implementations differ:

1. Projection layout differs.
MiniCPM-SALA Lightning uses one packed qkv projection and then splits q, k, v in minicpm.py. Other models may use separate q_proj and k_proj, or a different packed layout.

2. RoPE coverage differs.
Some models rotate the full head dimension. Some rotate only part of it. MiniCPM3 in this repo is a clear counterexample: it explicitly splits q into non-RoPE and RoPE parts, q_nope and q_pe, and handles them differently in minicpm3.py. That is not drop-in compatible with a kernel that assumes “rotate the whole head”.

3. q/k head structure differs.
Some models are MHA, some are GQA or MQA. If q heads and k heads have different packing or broadcasting rules, the fused kernel may need a different interface.

4. Norm placement differs.
Some models have no qk_norm at all. Some do q_norm and k_norm before RoPE. Some could do other variants. The fused kernel must match the exact order used by the model.

5. Rope style and position handling differ.
Different models may use different RoPE conventions, partial rotary dims, different position tensor shapes, or different in-place KV save assumptions.

6. Kernel contract differs.
A fused kernel may assume bf16 only, contiguous layout only, a certain head size, a certain rotary dimension, or a specific cache-save behavior. That is an implementation constraint, not a transformer-theory constraint.

So your intuition is right at the algorithm level, but not automatically right at the kernel-integration level.

What qk_norm_rope is in transformer terms

It corresponds to this part of attention:

Given hidden state h, first compute projections:
$$
q = W_q h,\quad k = W_k h,\quad v = W_v h
$$

Then apply RMSNorm separately to q and k:
$$
\hat q = \mathrm{RMSNorm}(q),\quad \hat k = \mathrm{RMSNorm}(k)
$$

A standard RMSNorm form is:
$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

Then apply RoPE to q and k at position p. RoPE acts on each 2D pair:
$$
\begin{bmatrix}
x'_{2j}\\
x'_{2j+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta_{p,j} & -\sin \theta_{p,j}\\
\sin \theta_{p,j} & \cos \theta_{p,j}
\end{bmatrix}
\begin{bmatrix}
x_{2j}\\
x_{2j+1}
\end{bmatrix}
$$

So:
$$
q^{(p)} = \mathrm{RoPE}(\hat q, p),\quad k^{(t)} = \mathrm{RoPE}(\hat k, t)
$$

Then the attention score is:
$$
s_{p,t} = \frac{q^{(p)} \cdot k^{(t)}}{\sqrt d}
$$

and finally:
$$
\mathrm{Attention}(p) = \sum_t \mathrm{softmax}(s_{p,t})\, v^{(t)}
$$

What “fused qk_norm_rope” means is simply: instead of doing

1. q_norm
2. k_norm
3. rope on q
4. rope on k

as separate memory-touching ops, one kernel does them together.

A concrete sample

Suppose one head has dimension 4, and we group it into two RoPE pairs:
$$
q = [1,2,3,4],\quad k = [2,1,0,2]
$$

After q_norm and k_norm, imagine we get:
$$
\hat q = [0.36, 0.73, 1.09, 1.46],\quad \hat k = [1.15, 0.58, 0, 1.15]
$$

Now suppose at one position the first pair uses angle \theta with:
$$
\cos\theta = 0,\quad \sin\theta = 1
$$

Then the first pair of q rotates from:
$$
[0.36, 0.73] \to [-0.73, 0.36]
$$

and the first pair of k rotates from:
$$
[1.15, 0.58] \to [-0.58, 1.15]
$$

The same happens for the second pair with its own angle. After that, those rotated q and k vectors are what go into the dot product for attention.

Why this matters for speed

The unfused path is logically:

1. project qkv
2. write q
3. write k
4. read q for q_norm
5. write q_norm
6. read k for k_norm
7. write k_norm
8. read q_norm for rope
9. write rope(q)
10. read k_norm for rope
11. write rope(k)

A fused path can collapse several of those reads and writes, which is why it can help.

What this means specifically for MiniCPM-SALA

For the normal MiniCPM hybrid/lightning path, MiniCPM-SALA does look structurally close enough to be a reasonable candidate, because it already has:

1. packed qkv projection
2. optional q_norm and k_norm
3. rope applied after q/k norm

all in minicpm.py.

So I am not saying “probably incompatible”. I am saying “promising, but must verify exact kernel contract before committing to a one-feature iteration”.

The strongest evidence that this is not always generic is MiniCPM3 in minicpm3.py, where q and k are not handled in the plain full-head RoPE style. That is exactly the sort of model-specific difference that breaks a naive reuse.

Short version

Your intuition is right at the high level: qk_norm_rope is a very standard attention subgraph.

My caution is only at the implementation level:
- standard math does not imply standard tensor layout
- standard math does not imply standard rotary-dim policy
- fused kernels usually encode stronger assumptions than the abstract algorithm

So MiniCPM-SALA is a plausible fusion target, but it still needs a compatibility check, not just a blind reuse.

If you want, next I can give you a very concrete proposal-only checklist for this direction:
1. what exact MiniCPM-SALA conditions we need to verify
2. what file/function would be the insertion point
3. what would make us reject this path quickly and switch to the Marlin branch instead
