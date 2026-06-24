Optimized tool selectionThe next useful step is to make the microtest fully diagnostic, not just “small.”

For this kernel, “tiny 1-warp microtest” should mean:

1. one warp only, 32 threads
2. one single `m16n8k32` MMA operation
3. no dequant complexity if possible for the first probe
4. values chosen so a wrong K mapping shows up immediately

A good first sample is a basis-style test.

Set A as mostly zeros, with a few rows selecting specific K positions:

```text
A is 16 x 32

row 0:  A[0,5]  = 1, all others 0
row 1:  A[1,13] = 1, all others 0
row 8:  A[8,21] = 1, all others 0
row 9:  A[9,29] = 1, all others 0

all other rows: all zeros
```

Set B so every K row is easy to recognize:

```text
B[k,n] = 10*k + n
```

so the first few B rows are:

```text
B[5,:]  = [50, 51, 52, 53, 54, 55, 56, 57]
B[13,:] = [130,131,132,133,134,135,136,137]
B[21,:] = [210,211,212,213,214,215,216,217]
B[29,:] = [290,291,292,293,294,295,296,297]
```

Then the correct output C = A x B must be:

```text
row 0: [50, 51, 52, 53, 54, 55, 56, 57]
row 1: [130,131,132,133,134,135,136,137]
row 8: [210,211,212,213,214,215,216,217]
row 9: [290,291,292,293,294,295,296,297]

all other rows: [0,0,0,0,0,0,0,0]
```

Why this is strong:

- if K is permuted, row 0 will return some other B row, not row 5
- if the upper/lower K half is swapped, rows using 21 or 29 will come out wrong
- if the 0..7 and 8..15 M halves are mixed, rows 0/1 vs 8/9 will move
- if column interpretation is wrong, the 8 output values inside a row will not match the expected sequence

So this test tells us much more than “pass/fail.” It tells us what kind of mapping is wrong.

After that, use a second sample with two nonzero K entries in one row, so you also test accumulation, not just selection:

```text
A row 0:
A[0,3]  = 1
A[0,19] = 2
all other A[0,k] = 0
```

Keep the same

```text
B[k,n] = 10*k + n
```

Then for row 0:

$$
C[0,n] = 1 \cdot B[3,n] + 2 \cdot B[19,n]
$$

Since

$$
B[3,n] = 30 + n,\quad B[19,n] = 190 + n
$$

we get

$$
C[0,n] = (30+n) + 2(190+n) = 410 + 3n
$$

So the expected row 0 is:

```text
[410, 413, 416, 419, 422, 425, 428, 431]
```

That second sample is useful because a pure one-hot row can only tell you “which K got selected.” The two-hot weighted row tells you whether the MMA path is also combining the intended K lanes correctly.

So the failure in the current test is:

- the kernel behaves correctly when A is too uniform to expose K-lane mistakes
- once A varies across K, the result diverges
- that strongly points to fragment packing or operand interpretation around the MMA path, not to a totally broken dequant

The reason I want this exact microtest before another kernel edit is that it turns the bug from “random activations fail” into something much sharper like:

- “K=5 is read as K=21”
- “rows 0..7 and 8..15 are swapped”
- “the 8 output columns are transposed/permuted”
- or “the accumulation mixes the wrong K quartets”

Explained the failure mode and laid out a concrete 1-warp diagnostic microtest with expected outputs that can pinpoint the remaining MMA fragment-mapping bug.Optimized tool selectionThe next useful step is to make the microtest fully diagnostic, not just “small.”

For this kernel, “tiny 1-warp microtest” should mean:

1. one warp only, 32 threads
2. one single `m16n8k32` MMA operation
3. no dequant complexity if possible for the first probe
4. values chosen so a wrong K mapping shows up immediately

A good first sample is a basis-style test.

Set A as mostly zeros, with a few rows selecting specific K positions:

```text
A is 16 x 32

row 0:  A[0,5]  = 1, all others 0
row 1:  A[1,13] = 1, all others 0
row 8:  A[8,21] = 1, all others 0
row 9:  A[9,29] = 1, all others 0

all other rows: all zeros
```

Set B so every K row is easy to recognize:

```text
B[k,n] = 10*k + n
```

so the first few B rows are:

```text
B[5,:]  = [50, 51, 52, 53, 54, 55, 56, 57]
B[13,:] = [130,131,132,133,134,135,136,137]
B[21,:] = [210,211,212,213,214,215,216,217]
B[29,:] = [290,291,292,293,294,295,296,297]
```

Then the correct output C = A x B must be:

```text
row 0: [50, 51, 52, 53, 54, 55, 56, 57]
row 1: [130,131,132,133,134,135,136,137]
row 8: [210,211,212,213,214,215,216,217]
row 9: [290,291,292,293,294,295,296,297]

all other rows: [0,0,0,0,0,0,0,0]
```

Why this is strong:

- if K is permuted, row 0 will return some other B row, not row 5
- if the upper/lower K half is swapped, rows using 21 or 29 will come out wrong
- if the 0..7 and 8..15 M halves are mixed, rows 0/1 vs 8/9 will move
- if column interpretation is wrong, the 8 output values inside a row will not match the expected sequence

So this test tells us much more than “pass/fail.” It tells us what kind of mapping is wrong.

After that, use a second sample with two nonzero K entries in one row, so you also test accumulation, not just selection:

```text
A row 0:
A[0,3]  = 1
A[0,19] = 2
all other A[0,k] = 0
```

Keep the same

```text
B[k,n] = 10*k + n
```

Then for row 0:

$$
C[0,n] = 1 \cdot B[3,n] + 2 \cdot B[19,n]
$$

Since

$$
B[3,n] = 30 + n,\quad B[19,n] = 190 + n
$$

we get

$$
C[0,n] = (30+n) + 2(190+n) = 410 + 3n
$$

So the expected row 0 is:

```text
[410, 413, 416, 419, 422, 425, 428, 431]
```

That second sample is useful because a pure one-hot row can only tell you “which K got selected.” The two-hot weighted row tells you whether the MMA path is also combining the intended K lanes correctly.

So the failure in the current test is:

- the kernel behaves correctly when A is too uniform to expose K-lane mistakes
- once A varies across K, the result diverges
- that strongly points to fragment packing or operand interpretation around the MMA path, not to a totally broken dequant

The reason I want this exact microtest before another kernel edit is that it turns the bug from “random activations fail” into something much sharper like:

- “K=5 is read as K=21”
- “rows 0..7 and 8..15 are swapped”
- “the 8 output columns are transposed/permuted”
- or “the accumulation mixes the wrong K quartets”

Explained the failure mode and laid out a concrete 1-warp diagnostic microtest with expected outputs that can pinpoint the remaining MMA fragment-mapping bug.
