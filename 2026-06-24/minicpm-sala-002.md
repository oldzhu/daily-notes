Compacted conversationThe current failure is not “the kernel crashes” or “dequant is completely broken.” The failure is more specific: the kernel only looks correct when the activation tile is too uniform to expose operand-order bugs.

In test_isolate.py, the pattern is:

1. simple weights + ones activations: pass
2. simple weights + random activations: fail
3. random weights + ones activations: pass
4. random weights + random activations: fail

That split matters. It says the remaining bug is very likely in the MMA operand path, not in the basic INT4 dequant itself. If dequant were fundamentally wrong, case 3 would also fail. Instead, the bad behavior appears when A varies across K. That is exactly what you would expect from a K-order mismatch, fragment-lane mapping bug, or wrong row/col interpretation for the FP8 operands fed into `mma.sync`.

Why the `ones` cases are misleading:
if one output element is
$$
C_{m,n}=\sum_{k=0}^{31} A_{m,k} B_{k,n},
$$
and `A_{m,k}=1` for every `k`, then
$$
C_{m,n}=\sum_{k=0}^{31} B_{k,n}.
$$
Now suppose the kernel accidentally permutes K before MMA:
$$
C'_{m,n}=\sum_{k=0}^{31} A_{m,k} B_{\pi(k),n}.
$$
If `A_{m,k}=1`, then
$$
C'_{m,n}=\sum_{k=0}^{31} B_{\pi(k),n}=\sum_{k=0}^{31} B_{k,n}.
$$
So a wrong K ordering can still produce the exact same result. That is why “all ones A” is a bad discriminator.

What I meant by a tiny 1-warp microtest is: strip everything down to one single `m16n8k32` MMA step, with data chosen so that every K position leaves a unique fingerprint in the output. Then any permutation or lane mixup becomes obvious immediately.

A good sample design is:

```text
Tile size:
A: 16 x 32
B: 32 x 8
C: 16 x 8
Only one warp
Only one MMA call
```

Use non-uniform, K-tagged A:
```text
A[m, k] = k + 1
```

or, even better, vary by row too:
```text
A[m, k] = 0.25 * (k + 1) + m
```

Use K-tagged and column-tagged B:
```text
B[k, n] = 10 * (n + 1) + (k + 1)
```

Then each output becomes:
$$
C_{m,n}=\sum_{k=0}^{31} A_{m,k}\,B_{k,n}
$$
and because both A and B depend on the exact K index, any wrong K permutation changes the answer a lot.

An even sharper version is a “basis” test. Pick one K at a time:

```text
A[0, k] = 1 only at k = 5, else 0
B[k, n] = unique values for each k,n
```

Then the correct result for row 0 is just the 5th row of B. If the kernel gives row 7 or row 21 of B instead, you know exactly that the K mapping is wrong, not just “numerically a bit off.”

So the purpose of the tiny microtest is not performance or realism. It is to answer one binary question cleanly:

- Does the warp feed the MMA instruction the exact intended `(m, k)` and `(k, n)` values?
- Or is there a hidden permutation/swizzle/layout mismatch that the `ones` tests were masking?

The practical takeaway is:
the current test failure means “random A exposes that our MMA input fragments are not representing the intended logical matrices,” even though the dequantized weights can look reasonable and the instruction itself is valid on SM120.

If you want, the next step I can do is write down one concrete 1-warp sample with expected numeric outputs for 2 or 3 chosen elements, so we can use it as a surgical truth table before touching the kernel again.
