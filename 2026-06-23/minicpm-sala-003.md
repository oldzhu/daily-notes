Let me check the diagnostic results that were written to the file:

Reviewed 2 files
Now I can see the error pattern clearly. Let me analyze:

Kernel vs FP8 ref: maxdiff=420, mean diff=70, only 2904/32768 outputs are even close (diff<10)
Errors are spread across all rows and columns — not isolated to specific rows/cols
The kernel output values are completely different from the reference
This confirms the kernel is computing wrong results for non-uniform weights. The error is not in FP8 quantization — it's a fundamental computation error.

Let me now think about what could cause this. The key clue: uniform weights pass, random weights fail. With uniform weights (all 0x77777777), every INT4 nibble is 7, so even if the kernel reads from the wrong position, it gets 7. With random weights, wrong position = wrong value.

The most likely cause: The kernel's SMEM→register data layout for the MMA fragments is wrong for the SM120 architecture.

Let me check if the PTX ISA for SM120's m16n8k32 has a different data layout than what I assumed:

Explore: Research SM120 MMA data layout
The data layout is confirmed correct. So the bug must be elsewhere. Let me think more carefully.

The uniform test passes but random fails. With uniform weights (all 0x77777777), every INT4 value is 7 → dequant to (7-1)*1 = 6 → FP8 0x4C. With random weights, different values.

Wait — I just realized something critical. The kernel uses __nv_fp8_e4m3(fv) to convert float to FP8. But what if fv is negative? FP8 e4m3fn can represent negative values. But the reference test has z4=1 and weights range 0-15, so fv = w4 - 1 ranges from -1 to 14. Negative values like -1 should be representable in FP8 e4m3fn.

Let me check: can the __nv_fp8_e4m3(float) constructor handle negative values correctly? And does mma.sync handle negative FP8 values?

Actually, let me focus on a different angle. The kernel does:


int kp = kg / 8, kbit = (kg % 8) * 4;int w4 = (qweight[kp * N + ng] >> kbit) & 0xF;
qweight shape is [K/8, N]. For K=128, K/8=16. So kp ranges 0..15. kp * N + ng = kp * 256 + ng. The max index is 15 * 256 + 255 = 4095. The tensor has 16 * 256 = 4096 elements. ✓

But wait — qweight is stored as a PyTorch tensor with shape [K/8, N]. PyTorch tensors are row-major by default. So qweight[kp, ng] is at byte offset (kp * stride(0) + ng * stride(1)) * sizeof(int32). With default strides, stride(0) = N = 256 and stride(1) = 1. So the linear index is kp * 256 + ng. ✓

OK, the indexing is correct. Let me look at this from a completely different angle. What if the issue is with how the MMA instruction handles the interaction between the A and B fragments?

The A fragment has 4 uint32_t registers (a_regs[0..3]), each holding 4 FP8 values. The B fragment has 2 uint32_t registers (b_regs[0..1]), each holding 4 FP8 values. The MMA computes D += A * B where A is [16, 32] and B is [32, 8] (since B is col-major, it's actually [8, 32] which means 8 rows × 32 cols in the GEMM sense).

Wait — row.col means A is row-major and B is column-major. In the GEMM C += A * B:

A is [16, 32] row-major (16 rows, 32 cols)
B is [8, 32] col-major (8 rows, 32 cols) — but wait, col-major means the K dimension is stored along rows...
Actually, let me reconsider. The PTX ISA mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 means:

A fragment: row-major, shape [16, 32]
B fragment: col-major, shape [32, 8] — 32 rows, 8 columns
So B is stored as 32 rows × 8 columns, with column-major layout. This means:

B[i][j] is at position [j][i] in the col-major storage
For thread t: b[0] = 4 bytes from row t/4, and b[1] = 4 bytes from row t/4 at offset +16
But in the kernel, W_fp8 stores weights as W_fp8[n][k] = [kTileN][kTileK] = [128][64]. The B fragment expects [32][8] (col-major). So B[ki][ni] = W_fp8[wn + ni][sk + ki].

For thread t: n_idx = wn + lane_id / 4 — this is the N index within the 8-wide N tile. k0 = sk + (lane_id % 4) * 4 — this is the K index.

But in col-major layout, the first dimension is K (32) and the second is N (8). So B[k][n] at col-major storage position k * 8 + n. For thread t, b[0] reads 4 bytes = 4 consecutive elements along the K dimension (since col-major stores columns contiguously).

So b[0] = B[k0..k0+3][n_idx] = W_fp8[n_idx][k0..k0+3] = 4 consecutive FP8 values at N-row n_idx, K-cols k0 to k0+3.

In the kernel: memcpy(&b_regs[0], &W_fp8[n_idx * kTileK + k0], sizeof(uint32_t)) — this reads 4 bytes from W_fp8[n_idx * 64 + k0]. That's 4 consecutive FP8 values at N-row n_idx, K-cols k0 to k0+3. ✓

And b[1] = B[k0+16..k0+19][n_idx] = W_fp8[n_idx][k0+16..k0+19] = 4 consecutive FP8 values at N-row n_idx, K-cols k0+16 to k0+19.

In the kernel: memcpy(&b_regs[1], &W_fp8[n_idx * kTileK + k1], sizeof(uint32_t)) where k1 = k0 + 16. ✓

So the B fragment reads 8 FP8 values from N-row n_idx, at K positions k0, k0+1, k0+2, k0+3, k0+16, k0+17, k0+18, k0+19. With k0 = sk + (lane_id % 4) * 4, the 4 threads in a row group cover K positions sk+0..sk+3, sk+4..sk+7, sk+8..sk+11, sk+12..sk+15 for the first 16, and sk+16..sk+19, sk+20..sk+23, sk+24..sk+27, sk+28..sk+31 for the second 16. Total: 32 K positions. ✓

Everything checks out. Let me look at this from yet another angle.

Actually, wait — I just realized something. The mma.sync instruction for m16n8k32 with FP8 e4m3 takes:

A: 4 uint32_t registers (4 × 4 bytes = 16 FP8 values)
B: 2 uint32_t registers (2 × 4 bytes = 8 FP8 values)
D: 4 float registers (4 × 4 bytes = 4 FP32 values)
But the A fragment has shape [16, 32] = 512 FP8 values. With 32 threads, each thread contributes 16 FP8 values = 4 uint32_t. 32 × 4 = 128 uint32_t = 512 FP8 values. ✓

The B fragment has shape [8, 32] (col-major) = 256 FP8 values. With 32 threads, each thread contributes 8 FP8 values = 2 uint32_t. 32 × 2 = 64 uint32_t = 256 FP8 values. ✓

The D fragment has shape [16, 8] = 128 FP32 values. With 32 threads, each thread contributes 4 FP32 values. 32 × 4 = 128 FP32 values. ✓

All dimensions check out. I'm really stuck. Let me try a completely different approach — write a minimal reproducer that checks the kernel output for a very small case where I can manually verify each step:

Created test_minimal.py and ran commands in directory
Very interesting! With N=8 (only one MMA N-tile), the kernel is perfectly correct!

The values are: 384, 512, 640, 768, 896, 1024, 1152, 1280. Let me verify: a = all 1.0, w_ref[k, n] = (k*N+n)%16 - 1. For n=0: w_ref[k, 0] = (k8+0)%16 - 1 = (8k)%16 - 1 = 0-1, 8-1, 0-1, 8-1, ... = -1, 7, -1, 7, ... for k=0,1,2,3,... Sum = 64(-1) + 64*7 = 384. ✓

Now let me test with N=16 (two N-tiles):

Created test_n_range.py and ran Python scripts
ALL PASS! The kernel works perfectly with this simple weight pattern (k*N+n)%16. But earlier with random weights it failed!

The difference: this test uses a = torch.ones(M, K) while the random test used a = torch.randn(M, K). Let me test with random activations:

Created test_isolate.py and ran Python scripts
Found it! The bug is in the activation handling, not the weight dequant!

Case 1 (simple weights, ones activations): PASS ✓
Case 2 (simple weights, random activations): FAIL ✗ (maxdiff=422)
Case 3 (random weights, ones activations): PASS ✓
Case 4 (random weights, random activations): FAIL ✗ (maxdiff=442)
The kernel handles weights correctly but fails when activations are non-uniform (random). The bug is in how the kernel reads activations from A_fp8_smem into the A fragment registers!

The activation load (Phase 2) writes to A_fp8_smem[m * kTileK + k]. The MMA A fragment reads from A_fp8_smem[row0 * kTileK + col0]. The indexing looks correct... but the issue must be in how the A fragment assembles the registers for the MMA instruction.

Wait — I tested with N=8 (single N-tile) and a=ones and it passed. But with a=random it fails. The A fragment reads the same way regardless of N. The only thing that changes with N is the number of N-tile iterations.

But actually, the A fragment doesn't depend on N at all — it reads from A_fp8_smem which is [kTileM][kTileK]. The N dimension only affects the B fragment (weights) and the epilogue.

So if the B fragment and epilogue are correct (proven by Case 1 and 3 passing), the bug must be in the A fragment assembly. But the A fragment assembly is the same code regardless of N...

Unless the bug is in how the MMA operates on the A and B fragments together. The MMA computes C += A * B where A is [16, 32] and B is [32, 8]. If A's values are all 1.0 (Case 1, 3), the MMA computes correctly regardless of B. But if A has varied values (Case 2, 4), the computation goes wrong.

This strongly suggests the A fragment values are wrong when activations are non-uniform. Let me check the A fragment assembly more carefully.

Actually, let me re-examine the A fragment. The A fragment reads:


int row0 = wm + lane_id / 4;int row1 = row0 + 8;int col0 = sk + (lane_id % 4) * 4;int col1 = col0 + 16;
For wm=0, lane_id=0: row0=0, row1=8, col0=sk, col1=sk+16
For wm=0, lane_id=4: row0=1, row1=9, col0=sk, col1=sk+16

Each thread reads 4 uint32_t = 16 FP8 values from A. With 32 threads, that's 512 FP8 values = 16 × 32 = the full A tile. ✓

But wait — the A fragment reads from A_fp8_smem[row * kTileK + col] where kTileK = 64. The A tile is [16, 32]. So the stride in SMEM is 64, but the A tile only has 32 columns. The remaining 32 columns per row are from the next K-tile iteration (sk=32).

For sk=0: reads col0=0,4,8,12 and col1=16,20,24,28. Total: 8 FP8 values per thread × 32 threads = 256 FP8 values = 16 × 16 = half of A.
For sk=32: reads col0=32,36,40,44 and col1=48,52,56,60. Total: another 256 FP8 values = 16 × 16 = the other half.

This is correct — the 32 K-columns of the A tile are covered by two sk iterations.

Hmm, but col0 = sk + (lane_id % 4) * 4. For sk=0, lane_id=0: col0=0. For sk=0, lane_id=1: col0=4. For sk=0, lane_id=2: col0=8. For sk=0, lane_id=3: col0=12. And col1 = col0 + 16 = 16, 20, 24, 28. So threads 0-3 read columns 0-3, 4-7, 8-11, 12-15 for the first 16, and 16-19, 20-23, 24-27, 28-31 for the second 16. Total: columns 0-31, 32 values. But each thread reads 8 FP8 values from a_regs[0]+a_regs[1] (first row) and 8 from a_regs[2]+a_regs[3] (second row). So 16 FP8 values per thread × 32 threads = 512 values = 16 × 32. ✓

Wait, I need to be more careful. Each thread reads:

a_regs[0] = 4 bytes from A[row0][col0..col0+3]
a_regs[1] = 4 bytes from A[row0][col1..col1+3]
a_regs[2] = 4 bytes from A[row1][col0..col1+3]
a_regs[3] = 4 bytes from A[row1][col1..col1+3]
Total: 16 bytes = 16 FP8 values per thread. 32 threads × 16 = 512 FP8 values. ✓

For the A tile [16, 32]:

16 rows × 32 cols = 512 FP8 values
Each thread contributes values from rows t/4 and t/4+8 (2 rows) and columns (t%4)*4..+3 and (t%4)*4+16..+3 (8 columns)
2 rows × 8 cols = 16 FP8 values per thread ✓
Everything checks out. I'm going to add a diagnostic printf to the kernel to compare the A fragment register values against expected values:

Read w4a8_fp8_qmma.cu, lines 80 to 130

Wait — I just noticed something crucial! Look at the loop structure:


for (int sk = 0; sk < kTileK; sk += kMmaK) {      // sk: 0, 32  for (int ms = 0; ms < 2; ++ms) {                  // ms: 0, 1    int wm = warp_m0 + ms * kMmaM;    for (int ns = 0; ns < 16; ++ns) {                // ns: 0..15      int wn = ns * kMmaN;      ...      // A fragment      int row0 = wm + lane_id / 4;      int row1 = row0 + 8;      int col0 = sk + (lane_id % 4) * 4;      int col1 = col0 + 16;      ...      // B fragment      int n_idx = wn + lane_id / 4;      int k0 = sk + (lane_id % 4) * 4;      int k1 = k0 + 16;      ...      asm("mma.sync ...");    }  }}
For each (sk, ms, ns) combination, the MMA operates on:

A tile: rows wm..wm+15, K-cols sk..sk+31
B tile: N-rows wn..wn+7, K-cols sk..sk+31
Output: rows wm..wm+15, N-cols wn..wn+7
The A fragment reads from A_fp8_smem[row * kTileK + col] where kTileK = 64. The A tile is [16, 32] but the SMEM stride is 64. So for sk=0, col0 ranges 0..12 and col1 ranges 16..28. For sk=32, col0 ranges 32..44 and col1 ranges 48..60.

This is correct! The A tile's 32 columns are read from SMEM positions 0..31 (for sk=0) and 32..63 (for sk=32). ✓

OK, I'm going to take a completely different approach. Let me check if the issue is with the __nv_fp8_e4m3 type and how it interacts with the MMA instruction. Maybe the MMA expects the FP8 values in a specific byte order within the uint32_t register.

Actually, let me check the byte order. The A fragment packs 4 FP8 values into a uint32_t. The PTX ISA says:

For row-major A: thread t holds a[0] = A[t/4][(t%4)*4..+4] — 4 consecutive FP8 values

"4 consecutive FP8 values" means 4 bytes. On a little-endian system (which CUDA uses), the first FP8 value is at the lowest byte address. So a_regs[0] as a uint32_t has:

Byte 0 (bits 0-7): A[row][col]
Byte 1 (bits 8-15): A[row][col+1]
Byte 2 (bits 16-23): A[row][col+2]
Byte 3 (bits 24-31): A[row][col+3]
The memcpy copies bytes from SMEM in order, so the byte order in the uint32_t matches the byte order in SMEM. This should be correct.

But wait — when we load a_regs[0] from SMEM, we do memcpy(&a_regs[0], &A_fp8_smem[row0 * kTileK + col0], 4). This copies 4 bytes from SMEM starting at address &A_fp8_smem[row0 * kTileK + col0]. On little-endian, the first byte goes to bits 0-7 of a_regs[0], the second to bits 8-15, etc.

The MMA instruction expects the uint32_t to contain 4 FP8 values in the correct byte order. Since both the SMEM and the register use the same byte ordering (little-endian), the values should be correct.

Hmm, let me try yet another angle. What if the issue is that the MMA instruction on SM120 requires specific alignment for the register values? Or what if there's a restriction on which registers can be used?

Actually, let me check something simpler. The asm volatile uses "r"(a_regs[0]) as input. But a_regs[0] is a uint32_t. The "r" constraint means a 32-bit register. For the MMA instruction, the A and B fragment inputs should be in 32-bit registers. This should be fine.

Wait — I just realized something. The MMA instruction on SM120 might require the input registers to be paired or have specific register alignment. Let me check if the "r" constraint for uint32_t works correctly with the MMA instruction.

Actually, the PTX ISA says the A and B fragment arguments should be .b32 (32-bit) registers. The "r" constraint in inline PTX ASM maps to a 32-bit register. This should be correct.

Let me try a completely different approach — use __hmma_m16n8k32_ld_a and __hmma_m16n8k32_ld_b built-in functions if they exist, or try a different register allocation:

Actually, let me check something I overlooked. The MMA instruction is:


mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
The last part f32 is the output type (D fragment = FP32). The e4m3.e4m3 means both A and B are FP8 e4m3. The row.col means A is row-major, B is column-major. The f32 before the types is the accumulation type.

Wait — is the accumulation type f32 (FP32) or is the output type f32? Let me check the PTX ISA syntax:


mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
Breaking it down:

m16n8k32: tile sizes M=16, N=8, K=32
row.col: A row-major, B col-major
f32: A element type = FP32? No, that doesn't make sense for FP8 MMA...
Actually, the PTX ISA format is:


mma.sync.aligned.m16n8k32.row.col.dtype.atype.btype
So: f32.e4m3.e4m3 = D type is f32, A type is e4m3, B type is e4m3. ✓

OK, the instruction format is correct. Let me try one more thing — test with M=16 (single MMA tile in M):

Created test_m_range.py and ran Python scripts
Right, the kernel requires M % 128 == 0. That's a constraint. Let me test with M=128 only but vary the random seed:

Created test_seeds.py and ran Python scripts
Consistently fails across all seeds. Now let me check one more thing — what if the bug is in how the kernel handles different K-tile iterations? Let me test with K=64 (single K tile):

Created test_k_range.py and ran Python scripts
Illegal memory access! For K=64 (single K-tile), the kernel crashes with illegal memory access. This confirms the bug is related to K-tile boundary conditions.

With K=64 and kTileK=64, the kernel runs one K-tile iteration. The dequant reads kg = kb + k where kb=0 and k ranges 0..63. kp = kg / 8 = 0..7, kbit = (kg % 8) * 4. qweight[kp * N + ng] — kp ranges 0..7, kp * N + ng max = 7*256+255 = 2047. qweight has K/8 * N = 8 * 256 = 2048 elements. ✓

Wait, but for K=64, qweight shape is [8, 256]. The kernel accesses qweight[kp * N + ng] where N=256. So kp * 256 + ng max = 7*256+255 = 2047. qweight has 8*256 = 2048 elements. Index 2047 is valid. ✓

But the A_fp8_smem load: A_fp8_smem[m * kTileK + k] where kTileK = 64. For m up to 127, m * 64 + k max = 127*64 + 63 = 8191. A_fp8_smem has kTileM * kTileK = 128 * 64 = 8192 elements. ✓

The activation load: a_fp8[mg * lda + kg] where lda = a_fp8.stride(0). For a with shape [128, 64], lda = 64. mg max = m0 + 127 = 127, kg max = kb + 63 = 63. 127 * 64 + 63 = 8191. a_fp8 has 128 * 64 = 8192 elements. ✓

Hmm, but wait — the kernel launch has grid dimensions dim3 grid(M / kTileM, ((int)N + kTileN - 1) / kTileN). For N=256, kTileN=128, the grid is (1, 2). For M=128, kTileM=128, grid is (1, 2). So there are 2 blocks: (mb=0, nb=0) and (mb=0, nb=1).

For block (0, 0): n0 = 0. ng = n0 + n where n ranges 0..127. Valid.
For block (0, 1): n0 = 128. ng = 128 + n where n ranges 0..127. ng max = 255. N = 256. ✓

But for K=64 and K=128: K/8 = 8 and 16. The qweight has K/8 rows. The kernel accesses qweight[kp * N + ng] where kp = kg / 8. For K=64, kg ranges 0..63, kp ranges 0..7. qweight has 8 rows. ✓

Wait — I'm not finding the illegal access from the code. Let me check with compute-sanitizer:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
226 memory errors! The address 0x102 is suspicious — it's a very low address, suggesting a null pointer dereference with a small offset. This is likely a misaligned SMEM read — reading from an address that's slightly off from the intended location.

Wait — 0x102 = 258 bytes from the start of some allocation. That's suspicious. Let me check — the SMEM is 16384 bytes. Address 0x102 from the start of SMEM would be byte 258. But the error says it's 139703206543102 bytes before the nearest allocation — that's a huge negative offset, meaning we're reading from an address that's way before any valid allocation.

This points to the A or B fragment reading from an out-of-bounds SMEM address. Let me check the fragment indexing more carefully for the K=64 case.

Actually, let me re-examine the SMEM allocation. W_fp8 has kTileN * kTileK = 128 * 64 = 8192 elements. A_fp8_smem = W_fp8 + 8192. So A_fp8_smem starts at byte 8192 within SMEM.

The A fragment reads A_fp8_smem[row0 * kTileK + col0]. For row0 = wm + lane_id / 4, wm = warp_m0 + ms * 16. warp_m0 = warp_id * (128/4) = warp_id * 32. So wm ranges from 0 to 96 (for warp 0, 1, 2, 3: 0, 32, 64, 96). row0 = wm + lane_id/4 max = 96 + 7 = 103. row1 = row0 + 8 max = 111.

A_fp8_smem[111 * 64 + 60 + 3] = A_fp8_smem[7167]. A_fp8_smem has 8192 elements. ✓

But wait — for the last block with mb = last, m0 = mb * 128. The global M is 128, so there's only one block in M. m0 = 0 for mb=0. The activation load checks mg < M && kg < K. For M=128, K=64, mg ranges 0..127 and kg ranges 0..63. All valid. ✓

Hmm, I can't find the out-of-bounds access statically. Let me check if the issue is with the SMEM size. kSmemBytes = kTileN * kTileK * 2 = 128 * 64 * 2 = 16384 bytes. The kernel allocates SMEM with <<<grid, block, kSmemBytes, stream>>>. This should be 16KB, which is within the 48KB shared memory limit per block on SM120. ✓

Wait — actually, let me check if the __nv_fp8_e4m3 type is 1 byte. Let me verify:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
OK, __nv_fp8_e4m3 is 1 byte. So SMEM size = 16384 bytes = 16KB. ✓

Let me check the compute-sanitizer output more carefully for the actual error location. The error address 0x102 is very small. Let me get more details:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Found it! The error is:

Invalid read of size 2 bytes (a bfloat16 read!)
at thread (64, 0, 0) in block (0, 0, 0)
Address 0x2 (very small, basically null+2)
Thread 64 is in warp 2 (64/32=2). The read of size 2 bytes (bfloat16) at address 0x2 is a null pointer dereference with offset 2.

This is a scales read! The kernel reads scales[gid * N + ng] where scales is a __nv_bfloat16*. If scales is a null pointer or gid * N + ng is out of bounds, we'd get this error.

Wait — for K=64, g = 128, so K/g = 0 (integer division: 64/128 = 0). So gid = kg / group_size = kg / 128 = 0 for all kg. gid * N + ng max = 0 * 256 + 255 = 255. scales has shape [K/g, N] = [0, 256] — a zero-size tensor!

That's the bug! For K < group_size, K/g = 0, so scales has shape [0, N] which is an empty tensor. The kernel tries to access scales[0 * N + ng] which is out of bounds because there are 0 rows!

But wait, this only happens for K=64 < g=128. For K=128, K/g = 1, so scales has shape [1, 256]. That should be valid.

But the earlier test with K=128 and random activations also failed... Let me check if the scales tensor is valid for that case:

For K=128, g=128: K/g = 1. scales shape = [1, 256]. gid = kg / 128 = 0 for all kg (since kg ranges 0..127). gid * N + ng max = 0 * 256 + 255 = 255. scales has 1 * 256 = 256 elements. Index 255 is valid. ✓

So the K=64 crash is a separate issue (scales tensor too small). But the K=128 random activation failure is a different bug.

Let me focus on the K=128 case. The compute-sanitizer didn't show errors for K=128 (the test ran without illegal memory access). So the bug is a logic error, not a memory error.

Let me now check the scales pointer more carefully. In the kernel launch:


static_cast<const __nv_bfloat16*>(scales.const_data_ptr())
For K=128, g=128, scales has 256 elements. This should work.

Wait — let me re-read the dequant code:


int gid = kg / group_size, z4 = 0;if (qzeros != nullptr) {    int zn = ng / 8, zb = (ng % 8) * 4;    z4 = ((qzeros[gid * (N / 8) + zn] >> zb) & 0xF) + 1;}float fv = ((float)w4 - (float)z4) * __bfloat162float(scales[gid * N + ng]);
gid = kg / group_size = kg / 128. For kg in 0..127, gid = 0. scales[0 * N + ng] = scales[ng]. For ng in 0..255, this reads scales[0..255]. scales has 256 elements. ✓

OK, the K=128 case doesn't have a memory error. But the results are wrong. Let me think about what could cause wrong results without memory errors...

Actually, I just realized I should check the compute-sanitizer output for the K=128 case too. Let me run it:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Zero memory errors! The K=128 case has no memory errors — it's a pure logic bug. The kernel computes wrong results without accessing invalid memory.

Let me think about this more carefully. Cases 1 and 3 pass (ones activations, any weights). Cases 2 and 4 fail (random activations). The A fragment reads from A_fp8_smem which is loaded from a_fp8[mg * lda + kg].

Wait — a_fp8 is a const __nv_fp8_e4m3*. The activation tensor has shape [M, K] with M=128, K=128. lda = a_fp8.stride(0). For a contiguous row-major tensor, lda = K = 128. So a_fp8[mg * lda + kg] = a_fp8[mg * 128 + kg]. For mg max 127 and kg max 127: 127 * 128 + 127 = 16383. a_fp8 has 128 * 128 = 16384 elements. ✓

But wait — the lda parameter is passed from the C++ wrapper:


(int)a_fp8.stride(0), (int)c_bf16.stride(0)
For a tensor with shape [128, 128] that's contiguous, stride(0) = 128. ✓

Hmm, let me check the A_fp8_smem load more carefully:


for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {    int m = i / kTileK, k = i % kTileK;    int mg = m0 + m, kg = kb + k;    A_fp8_smem[m * kTileK + k] =        (mg < M && kg < K) ? a_fp8[mg * lda + kg] : __nv_fp8_e4m3{0};}
For K=128, kTileK=64, kb=0: kg = k ranges 0..63. a_fp8[mg * 128 + kg] — kg is the local K index within the tile. But a_fp8 is indexed by global K index kg. Since kb=0, kg = k which is both the local and global K index. ✓

For kb=64: kg = 64 + k where k ranges 0..63. a_fp8[mg * 128 + (64 + k)] — this reads from global K index 64..127. ✓

OK, the activation load is correct. Let me now check if the A fragment reads from the correct SMEM positions during the MMA phase.

The A fragment reads A_fp8_smem[row0 * kTileK + col0]. kTileK = 64. For row0 = wm + lane_id/4 and col0 = sk + (lane_id%4)*4:

For wm=0, lane_id=0, sk=0: row0=0, col0=0. Reads A_fp8_smem[0 * 64 + 0] = A_fp8_smem[0]. This was loaded by thread i=0 in Phase 2: m=0/64=0, k=0%64=0. The value is a_fp8[m0*lda + kb + k] = a_fp8[0*128 + 0 + 0] = a_fp8[0]. ✓

For wm=32, lane_id=0, sk=0 (warp 1): row0=32, col0=0. Reads A_fp8_smem[32 * 64 + 0] = A_fp8_smem[2048]. Loaded by thread i=2048: m=2048/64=32, k=2048%64=0. Value = a_fp8[32*128 + 0] = a_fp8[4096]. ✓

Everything checks out. I'm completely stuck on the logic error. Let me try a completely different approach — add a diagnostic to the kernel that prints the A fragment register values:

Updated w4a8_fp8_qmma.cu and ran commands
The row0 and col0 are defined inside the inner scope. Let me fix:

Updated w4a8_fp8_qmma.cu and ran commands
The A fragment values look reasonable (FP8 e4m3 values like 1.375, -0.125, etc.). Let me verify one: a_regs[0]=0xbb2a343b. In little-endian, the bytes are 0x3b, 0x34, 0x2a, 0xbb. The first FP8 value is 0x3b = sign=0, exp=7 (bias 7 = 2^0=1), mantissa=0b011 = 3/8=0.375, so value = 1.375. ✓ That matches the printed value.

Now let me check: do these values match the input tensor? a[0, 0] should be the first element. Let me verify:

Created test_verify_a.py and ran Python scripts
Now let me compare with the diagnostic output. For tid=0: the kernel printed a_regs[0]=0xbb2a343b(1.375). But the expected value is a_regs[0]=0x34b5233b. These are completely different!

0xbb2a343b vs 0x34b5233b — not even close!

Wait — the kernel diagnostic was run with a different random seed (random.seed(42) before the kernel call). Let me check if the verify script also uses the same seed. Looking at the verify script: it uses random.seed(42) which is the same seed. But the values don't match!

Oh wait — the diagnostic was from a separate kernel run than the verify script. The random seed 42 is used in both, but the torch.randn generates different values on GPU than CPU... Actually, both are on GPU. Let me check.

Actually, looking more carefully: the diagnostic output shows a_regs[0]=0xbb2a343b(1.375) for tid=0. The verify script shows a_regs[0]=0x34b5233b for tid=0. These are different.

Wait — the diagnostic kernel prints the A fragment values during the MMA phase (after SMEM load). The verify script computes the expected values directly from the tensor. If the SMEM load is correct, they should match. But they don't!

This means the A_fp8_smem load (Phase 2) is loading wrong values, OR the A fragment read from SMEM is reading from the wrong position.

Let me compare more carefully. The diagnostic output for tid=0: row0=0 col0=0 a_regs[0]=0xbb2a343b(1.375). This reads from A_fp8_smem[0]. The expected value at A_fp8_smem[0] should be a_fp8[0, 0] = a[0, 0] in FP8.

From the verify script: a[0, 0] = 1.375 (first value in tid=0's vals0 list). And the diagnostic also shows 1.375. So the value is correct!

But the hex values differ: 0xbb2a343b vs 0x34b5233b. Let me check: the diagnostic shows a_regs[0]=0xbb2a343b(1.375). The verify shows a_regs[0]=0x34b5233b vals=[1.375, 0.171875, -0.8125, 0.75].

These are different hex values! But both claim the first FP8 value is 1.375. Let me check: 0xbb2a343b in little-endian bytes: 3b, 34, 2a, bb. First byte 0x3b → FP8 e4m3: sign=0, exp=7 (bias 7 = 2^0=1), mantissa=0b011 = 3/8 = 0.375 → value = 1.375. ✓

0x34b5233b in little-endian bytes: 3b, 23, b5, 34. First byte 0x3b → FP8 e4m3: same thing, 1.375. ✓

So the first FP8 value is the same (1.375 = 0x3b), but the remaining 3 FP8 values differ! The diagnostic reads 0xbb2a343b which is bytes [0x3b, 0x34, 0x2a, 0xbb] = FP8 values [1.375, 0.75, -0.3125, -1.375]. But the verify expects 0x34b5233b = bytes [0x3b, 0x23, 0xb5, 0x34] = FP8 values [1.375, 0.171875, -0.8125, 0.75].

So the first byte matches but the next 3 bytes differ! This means the kernel is reading from the wrong SMEM position — it's shifted by some amount.

Wait — 0xbb2a343b has bytes [0x3b, 0x34, 0x2a, 0xbb]. The expected 0x34b5233b has bytes [0x3b, 0x23, 0xb5, 0x34]. The first byte matches (0x3b = 1.375). But the rest doesn't.

Hmm, but the diagnostic and verify scripts use different random seeds! The diagnostic was from the earlier test_isolate.py run, and the verify script is a separate run. Even though both use random.seed(42), the sequence of random numbers depends on all the calls made before. Let me make them use the exact same setup.

Actually, let me just add the diagnostic printf to also print the expected values, so I can compare directly in one run:

Created test_compare_a.py and ran Python scripts
They match! The diagnostic output and expected values are identical:

DIAG: tid=0 row0=0 col0=0 a_regs[0]=0x36a34124(0.1875)

Expected: tid=0 row=0 col0=0 exp_a_regs[0]=0x36a34124 vals=[0.1875, ...] ✓

DIAG: tid=0 a_regs[1]=0x913533b3(-0.6875)

Expected: tid=0 col1=16 exp_a_regs[1]=0x913533b3 vals=[-0.6875, ...] ✓

So the A fragment is loaded correctly! The values match perfectly. The bug must be elsewhere.

Let me check the B fragment and the MMA computation itself:

Updated w4a8_fp8_qmma.cu and ran commands
cp is defined after the diagnostic block. Let me move the diagnostic after cp:

Updated w4a8_fp8_qmma.cu and ran commands
Now I can see both the first iteration (sk=0) and second iteration (sk=32) of the MMA. The A and B fragment values look correct. But the c values are all 0 (since this is the first MMA call).

Let me now verify the B fragment values against what the reference expects. For tid=0, n_idx=0, k0=0: B[0][0..3] = W_fp8[0][0..3]. The B fragment reads 4 FP8 values from W_fp8[0 * 64 + 0] = W_fp8[0..3]. The hex value is 0x50385540. In little-endian bytes: 0x40, 0x55, 0x38, 0x50. As FP8 e4m3: 0x40=2.0, 0x55=10.0, 0x38=1.0, 0x50=8.0. The printed value is 2 (first byte only).

Wait — the first byte 0x40 = FP8 e4m3: sign=0, exp=8 (bias 7 = 2^1=2), mantissa=0 → value = 2.0. ✓

But the reference weight for w[0,0] should be w4 - 1 where w4 is the INT4 value at position (k=0, n=0). Let me check what w4 is:

From the test_compare_a.py script, random.seed(42) is used. Let me check what the reference weight is for (k=0, n=0):

The reference computed w_ref[0, 0] = w4 - 1 where w4 is extracted from qw[0, 0]. The B fragment reads from W_fp8[n_idx=0][k0=0..3] = dequantized weights for (k=0..3, n=0). The first FP8 value is 2.0. So w_ref[0, 0] = 2.0. This means w4 = 3.0, so the INT4 value at position (k=0, n=0) is 3. That seems correct for the random seed.

Now the key question: does the MMA compute the correct result? The A and B fragment values look correct. The MMA should compute D += A * B where each thread contributes to 4 output positions.

Let me check the final accumulator values after the MMA completes:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
The diagnostic only fires for sk=0 (first K-tile). The sk loop goes sk=0, 32, but the diagnostic only runs once because it checks sk == 0. And it only fires for kb=0, ms=0, ns=0. So we only see the first MMA call.

The output shows c={0,0,0,0} which is correct — accumulators start at 0 before the first MMA.

The diagnostic doesn't show results AFTER the MMA because the printf is before the asm volatile. I need to add a post-MMA diagnostic. But given the time, let me take a different approach.

Instead of more diagnostics, let me look at the actual MMA instruction more carefully. The key realization from the data is:

A fragment values are correct (verified against expected)
B fragment values look reasonable (FP8 values like 2, 6, 10, etc.)
The MMA produces wrong results (verified by the output comparison)
The only remaining possibility is that the MMA instruction itself has a problem. Let me check if the inline PTX assembly is correct for SM120.

Actually, wait — I just realized something. The mma.sync instruction on SM120 might require a different syntax than what I'm using. Let me check the CUDA 12.x PTX ISA for SM120:

The instruction is:


mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
But maybe on SM120, the FP8 MMA uses a different instruction name? Let me check if it should be mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 or something else.

Actually, let me check if the issue is that SM120 requires tcgen05 instructions instead of mma.sync. The conversation summary mentioned "tcgen05 is SM100-only" but that might be wrong for SM120.

Or maybe the issue is simpler — maybe the "r" constraint for uint32_t inputs doesn't work correctly with the MMA instruction, and I need to use "h" (16-bit) or some other constraint.

Let me try a different approach. Instead of inline PTX, let me check if CUDA provides a built-in for this MMA operation:

Searched for text mma.sync (**/sgl-kernel/**), 14 results
Let me check the qserve_w4a8_per_chn_gemm.cu which uses m16n8k32 with int8 — it might have a working pattern:

Read qserve_w4a8_per_chn_gemm.cu, lines 100 to 150

I see a key difference! In the qserve_w4a8_per_chn_gemm.cu, the mma_m16n8k32 function uses "=r" for the output (C fragment) and "r" for the input (A, B fragments, and C fragment for accumulation). The output uses output constraints "=r" not "+r".

But in our kernel, we use "+f"(cp[0]) for the C fragment. The +f is a read-write constraint for a float register. But the MMA instruction outputs f32 values...

Wait — actually the key difference is that qserve uses s32 outputs while we use f32. The "=r" vs "+f" difference is because qserve uses s32 (int accumulator) while we use f32 (float accumulator).

But hold on — let me look at the output constraint more carefully. In qserve:


: "=r"(((int*)C_warp)[0]), "=r"(((int*)C_warp)[1]), ...: "r"(((unsigned*)A_shared_warp)[0]), ...  "r"(((int*)C_warp)[0]), "r"(((int*)C_warp)[1]), ...
The C fragment appears in BOTH the output (with "=r") and input (with "r") lists. This is because the MMA instruction reads the old accumulator value and writes the new accumulated value. The "=r" tells the compiler the output register is written, and the "r" tells it the input register is read.

In our kernel:


: "+f"(cp[0]), "+f"(cp[1]), "+f"(cp[2]), "+f"(cp[3]): "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),  "r"(b_regs[0]), "r"(b_regs[1]),  "f"(cp[0]), "f"(cp[1]), "f"(cp[2]), "f"(cp[3])
We use "+f" for the output (read-write) and also pass them as "f" inputs. This is equivalent to what qserve does with "=r" + "r".

But wait — there's a subtle issue! The "f" constraint means a float register (32-bit). The "r" constraint means a general-purpose register (32-bit). For the MMA instruction, the C/D fragment operands should be in float registers (since they're f32 values). The A and B fragment operands should be in general-purpose registers (since they're uint32_t values containing packed FP8 bytes).

Our code uses:

C/D: "+f"(cp[0]) — float register, read-write ✓
A: "r"(a_regs[0]) — GPR ✓
B: "r"(b_regs[0]) — GPR ✓
C/D input: "f"(cp[0]) — float register ✓
This looks correct. Let me check if there's an issue with the cp pointer type. cp is float* pointing to c_regs[ms][ns]. The c_regs array is float[2][16][4]. The cp[0..3] are the 4 float values for this MMA call.

Hmm, but cp points to c_regs[ms][ns] which is a float[4]. The values are initialized to 0.0 and accumulated across MMA calls. This should be correct.

OK, I'm going to try one more thing. Let me check if the issue is with the cp values being used as both input and output in the same asm statement. The "+f"(cp[0]) tells the compiler that cp[0] is both read and written. But we also pass "f"(cp[0]) as a separate input operand. This creates a duplicate operand situation which might confuse the compiler.

Let me try using the qserve pattern (separate output and input operands):

Updated w4a8_fp8_qmma.cu and ran commands
Still failing. The "+f" → "=f" + store change didn't fix it. Let me think about this differently.

The A and B fragment values are correct. The MMA instruction is correct. But the output is wrong when activations are non-uniform.

Wait — I just realized something. Let me check: when a = ones(M, K), the A fragment values are all 1.0 (FP8 0x38). When a = randn(M, K), the A fragment values vary. The MMA computes D += A * B. If A values are all 1.0, then D += 1.0 * B = sum(B). If A values vary, then D += A[i] * B[i].

The key insight: with a = ones, every A fragment register is 0x38383838 (four 1.0 FP8 values). With random activations, the A fragment registers have varied values.

What if the MMA instruction on SM120 has a restriction on the A fragment register values? For example, what if it requires the A fragment to contain only values in a specific range, or what if it doesn't handle certain FP8 bit patterns correctly?

Actually, let me check something else. The __nv_fp8_e4m3 type — is it the same as __nv_fp8_e4m3fn? The e4m3 format has two variants: e4m3fn (finite-only, no NaN/inf) and e4m3 (with NaN). The MMA instruction specifies e4m3 which might be the variant with NaN support.

Let me check if the __nv_fp8_e4m3 constructor produces NaN bit patterns for certain float values:

Executed command in terminal
OK, the FP8 conversions look correct. 0x7e is the max value (448.0). No NaN bit patterns.

Let me now think about this problem from a completely different angle. The key observation is:

Weights correct ✓ (Cases 1, 3 pass)
Activations correct ✓ (A fragment values verified)
B fragment correct ✓ (verified against reference)
MMA instruction correct ✓ (verified syntax)
Output wrong ✗ (Cases 2, 4 fail with random activations)
The only thing left to check is: does the MMA instruction on SM120 actually support m16n8k32 with FP8 e4m3? Maybe this instruction is not available on SM120 and is silently emulated or produces wrong results.

Let me check the CUDA compute capability for SM120:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Wait — the device is "NVIDIA RTX 6000D" with Compute Capability 12.0 and 156 SMs!

This is NOT the expected SM120 Blackwell GPU! The SM120_RTX_PRO_HARDWARE.md says:

96 SMs for RTX PRO 6000
But this device has 156 SMs and is called "RTX 6000D" (not "RTX PRO 6000")
The compute capability is 12.0 which is SM120. But the GPU name is different.

Actually, wait — 156 SMs × 128 CUDA cores/SM = 19968 CUDA cores. That could be a different Blackwell variant.

But the key point is: SM 12.0 with -arch=sm_120a. Let me check if sm_120a is the correct architecture flag. The a suffix enables sm_90a features (like TMA, etc.). For SM120, the a suffix might be different.

Actually, let me check if m16n8k32 with e4m3 is actually supported on SM120. Let me look at the CUDA PTX ISA documentation:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
CUDA 12.8. Let me check if the MMA instruction is correctly compiled:

Executed command in terminal
The MMA instruction itself is broken on this SM120 GPU!

Even with the simplest test (all A=1.0, all B=2.0), only thread 0 gets non-zero values (96.0 instead of 64.0), and all other threads get 0!

Wait — the SMEM layout is wrong. The A fragment expects A[16][32] but I only allocated smem_a[256] with stride 32. The A fragment reads smem_a[row0 * 32 + col0]. For row0=0, col0=0: reads smem_a[0]. For row0=7, col0=12: reads smem_a[7*32+12] = smem_a[236]. smem_a has 256 elements. ✓

But the SMEM is initialized with smem_a[tid] = A[tid] for tid < 256. The elements of A are all 1.0. So all 256 elements of smem_a are 1.0. ✓

Similarly for B: all 256 elements are 2.0. ✓

With A=1.0 (FP8 0x38) and B=2.0 (FP8 0x40), the MMA should compute:

Each output position = sum over 32 K elements of 1.0 * 2.0 = 64.0
But only thread 0 gets 96.0 and all others get 0. This is completely wrong!

The SMEM layout for A is wrong! The A tile is [16, 32] but I'm using stride 32. The correct stride should be... well, the A tile is exactly 32 columns wide, so stride 32 is correct for a contiguous tile.

Wait — the issue is that B is col-major with shape [8, 32] but I stored it as [8, 32] row-major in SMEM. For col-major B, B[k][n] is stored at position k * 8 + n. So B[0][0] is at index 0, B[1][0] is at index 8, etc.

But in my test, I stored B as row-major: smem_b[tid] = B[tid] where B is a flat array. The B fragment reads smem_b[n_idx * 32 + k0]. If B is stored row-major [8][32], then B[n][k] is at n * 32 + k. But col-major B[k][n] would be at k * 8 + n.

The MMA instruction expects B in col-major layout: B[k][n] at position k * 8 + n. But my test stores B as row-major B[n][k] at position n * 32 + k. So the B fragment reads from the wrong positions!

That's the bug in my test, not in the MMA instruction. Let me fix the test:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
The MMA instruction works correctly on SM120! All threads get 64.0 as expected.

So the issue is in how our kernel stores B in SMEM. The key difference: in my test, B is stored in col-major format: smem_b[k * 8 + n]. But in our kernel, B is stored as W_fp8[n * kTileK + k] — which is row-major [kTileN][kTileK]!

The MMA instruction expects B in col-major layout: B[k][n] at position k * 8 + n. But our kernel stores weights in row-major W_fp8[n][k] at position n * kTileK + k. The B fragment reads W_fp8[n_idx * kTileK + k0] which reads from W_fp8[n_idx][k0..k0+3] — this is 4 consecutive FP8 values at row n_idx, columns k0..k0+3.

But in col-major layout, the B fragment expects 4 consecutive FP8 values at column n_idx, rows k0..k0+3. So B[k0..k0+3][n_idx] should be at positions k0 * 8 + n_idx, (k0+1) * 8 + n_idx, (k0+2) * 8 + n_idx, (k0+3) * 8 + n_idx.

But our kernel reads W_fp8[n_idx * 64 + k0], W_fp8[n_idx * 64 + k0 + 1], ... which are positions n_idx * 64 + k0, n_idx * 64 + k0 + 1, .... These are 4 consecutive values along the K dimension at a fixed N index. In col-major layout, this should be 4 consecutive values along the K dimension at a fixed N index — which IS what col-major B[k][n] means: consecutive K values at the same N.

Wait — let me re-examine. In my test, I stored B as smem_b[k * 8 + n] where k is the K index (0..31) and n is the N index (0..7). The B fragment reads smem_b[k0 * 8 + n_idx] for b_regs[0] and smem_b[k1 * 8 + n_idx] for b_regs[1]. These are 4 consecutive bytes at K positions k0..k0+3 and k1..k1+3, all at the same N index n_idx.

In our kernel, W_fp8 is stored as W_fp8[n * kTileK + k] = W_fp8[n * 64 + k]. The B fragment reads W_fp8[n_idx * 64 + k0] = W_fp8[n_idx * 64 + k0]. This reads 4 consecutive bytes at K positions k0..k0+3 at N index n_idx. This is the same as smem_b[k0 * 8 + n_idx] if kTileK = 8.

But kTileK = 64, not 8! The B fragment expects the K stride to be 8 (the N dimension size), but our SMEM has K stride 64.

Actually, wait — the B fragment doesn't care about the stride. It just reads 4 consecutive bytes starting at a given address. The address &W_fp8[n_idx * 64 + k0] points to the 4 FP8 values at W_fp8[n_idx][k0], W_fp8[n_idx][k0+1], W_fp8[n_idx][k0+2], W_fp8[n_idx][k0+3]. These are 4 consecutive values in the K dimension at N index n_idx. In col-major layout B[k][n], these correspond to B[k0][n_idx], B[k0+1][n_idx], B[k0+2][n_idx], B[k0+3][n_idx] which are 4 consecutive K values at the same N. ✓

So the B fragment reads the correct values. The issue must be elsewhere.

Let me compare the B fragment read more carefully between my test and the kernel:

My test (working):


memcpy(&b_regs[0], &smem_b[k0 * 8 + n_idx], 4);  // B[k0..k0+3][n_idx]memcpy(&b_regs[1], &smem_b[k1 * 8 + n_idx], 4);  // B[k1..k1+3][n_idx]
Our kernel:


memcpy(&b_regs[0], &W_fp8[n_idx * kTileK + k0], 4);  // W[n_idx][k0..k0+3]memcpy(&b_regs[1], &W_fp8[n_idx * kTileK + k1], 4);  // W[n_idx][k1..k1+3]
In my test: smem_b[k0 * 8 + n_idx] = B at K-index k0, N-index n_idx.
In our kernel: W_fp8[n_idx * 64 + k0] = W at N-index n_idx, K-index k0.

The values are the same (weight at (n_idx, k0)), just transposed in storage. The memcpy reads 4 consecutive bytes from the address. In my test, the 4 bytes are at K positions k0, k0+1, k0+2, k0+3 (since stride is 8, k0*8 + n_idx to (k0+3)*8 + n_idx). In our kernel, the 4 bytes are at K positions k0, k0+1, k0+2, k0+3 (since stride is 1 in the K dimension).

Both read 4 consecutive FP8 values at the same N index, covering K positions k0..k0+3. ✓

So the B fragment read is correct. The issue must be elsewhere.

Actually, let me re-examine my test more carefully. In my test, the SMEM for B has shape [32][8] (col-major, 32 K × 8 N). Each thread reads b_regs[0] from smem_b[k0 * 8 + n_idx] — this is 4 consecutive bytes at addresses k0*8+n_idx, k0*8+n_idx+1, k0*8+n_idx+2, k0*8+n_idx+3. These are at K-index k0, k0, k0, k0 and N-index n_idx, n_idx+1, n_idx+2, n_idx+3!

Wait, that's wrong! In col-major, smem_b[k * 8 + n] = B[k][n]. The 4 consecutive bytes starting at smem_b[k0 * 8 + n_idx] are:

smem_b[k0 * 8 + n_idx + 0] = B[k0][n_idx+0]
smem_b[k0 * 8 + n_idx + 1] = B[k0][n_idx+1]
smem_b[k0 * 8 + n_idx + 2] = B[k0][n_idx+2]
smem_b[k0 * 8 + n_idx + 3] = B[k0][n_idx+3]
These are 4 consecutive values in the N dimension at the same K index! Not 4 consecutive K values!

But the MMA spec says: b[0] = B[t/4][(t%4)*4..+4] — 4 consecutive values along the K dimension (second index for col-major B).

So the 4 bytes in b_regs[0] should be 4 consecutive K values at the same N index. But my test reads 4 consecutive N values at the same K index!

But my test PASSES! So either:

The MMA instruction actually expects N-consecutive values (not K-consecutive), OR
My test has a coincidental pass because all values are the same (2.0)
Let me check with non-uniform B values:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
The MMA output is wrong even with non-uniform B values! The values don't match expectations. But the uniform test (all B=2.0) passed coincidentally because all values were the same.

Wait — the expected values are wrong too. Let me recalculate. With A=1.0 for all elements and B[n][k] = n*10 + k:

D[row][col] = sum_{k=0}^{31} A[row][k] * B[col][k] = sum_{k=0}^{31} 1.0 * (col10 + k) = 32col10 + sum_{k=0}^{31} k = 320col + 496

For col=0: 496. But the MMA gives 926 for D[0][0]. That's wrong.

But wait — the B fragment reads b_regs[0] from smem_b[k0 * 8 + n_idx] which is 4 consecutive bytes starting at B[k0][n_idx]. These 4 bytes are at positions:

k0*8 + n_idx + 0 = B[k0][n_idx+0]
k0*8 + n_idx + 1 = B[k0][n_idx+1]
k0*8 + n_idx + 2 = B[k0][n_idx+2]
k0*8 + n_idx + 3 = B[k0][n_idx+3]
These are 4 consecutive values in the N dimension, not the K dimension!

But the MMA spec says b[0] should be 4 consecutive values along the K dimension (since B is col-major, the K dimension is the inner/contiguous dimension).

So the correct read should be: b_regs[0] = 4 consecutive K values at the same N = B[k0][n_idx], B[k0+1][n_idx], B[k0+2][n_idx], B[k0+3][n_idx].

In col-major storage, these are at positions:

k0*8 + n_idx = B[k0][n_idx]
(k0+1)*8 + n_idx = B[k0+1][n_idx]
(k0+2)*8 + n_idx = B[k0+2][n_idx]
(k0+3)*8 + n_idx = B[k0+3][n_idx]
These are NOT consecutive in memory! They are spaced 8 bytes apart (stride 8 in the K dimension).

So memcpy(&b_regs[0], &smem_b[k0 * 8 + n_idx], 4) reads 4 consecutive bytes which are B[k0][n_idx..n_idx+3] — 4 values in the N dimension. But the MMA expects 4 values in the K dimension: B[k0..k0+3][n_idx].

The correct read should be 4 individual byte reads:


uint8_t b_bytes[4] = {    smem_b[(k0+0) * 8 + n_idx],    smem_b[(k0+1) * 8 + n_idx],    smem_b[(k0+2) * 8 + n_idx],    smem_b[(k0+3) * 8 + n_idx]};memcpy(&b_regs[0], b_bytes, 4);
This is a gather operation, not a contiguous read!

So the bug in our kernel is that we read 4 consecutive bytes from SMEM for b_regs[0], but the MMA expects 4 values from 4 different K positions (which are not contiguous in col-major storage).

This means our entire B fragment assembly is wrong! Let me fix it:

Executed command and reviewed 3 files
The gather-based approach also gives wrong results (value=1 instead of 496). The B fragment values are completely wrong. Let me re-examine the PTX ISA specification more carefully.

Actually, I think my understanding of the B fragment layout is wrong. Let me reconsider.

The PTX ISA says for mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32:

A: row-major, shape [16, 32], each thread provides 4 uint32_t
B: col-major, shape [32, 8], each thread provides 2 uint32_t
Wait — "col-major" means the first index is the column and the second is the row. So B has 32 columns and 8 rows (in the GEMM sense). But for the MMA, A is [M, K] and B is [K, N]. So:

A: [16, 32], row-major → M=16, K=32
B: [8, 32], col-major → K=32, N=8
In col-major storage, B[k][n] is stored with k as the inner dimension. So B[k][n] is at position k * 8 + n. The 4 consecutive bytes starting at B[k0 * 8 + n_idx] are:

B[k0][n_idx], B[k0][n_idx+1], B[k0][n_idx+2], B[k0][n_idx+3]
These are 4 consecutive values in the N dimension at the same K index.

But the PTX ISA says b[0] = B[t/4][(t%4)*4..+4]. The first index is t/4 = the K index (since B is col-major, first index = K). The second index is (t%4)*4 = the N index. So b[0] = 4 consecutive bytes at B[K=t/4][N=(t%4)*4..+4].

In col-major storage, B[k][n] is at k*8 + n. So B[t/4][(t%4)*4..+4] is at positions:

(t/4)*8 + (t%4)*4 + 0 = (t/4)*8 + (t%4)*4
(t/4)*8 + (t%4)*4 + 1
(t/4)*8 + (t%4)*4 + 2
(t/4)*8 + (t%4)*4 + 3
These ARE 4 consecutive bytes at K-index t/4, N-indices (t%4)*4 to (t%4)*4+3.

So memcpy(&b_regs[0], &smem_b[(t/4)*8 + (t%4)*4], 4) is the correct read!

In the kernel: n_idx = wn + lane_id / 4 = N index, k0 = sk + (lane_id % 4) * 4 = K offset.

Wait — the PTX ISA says b[0] = B[t/4][(t%4)*4..+4]. Here t/4 is the first index (K for col-major B) and (t%4)*4 is the second index (N for col-major B).

So in the kernel, n_idx = wn + lane_id / 4 should be the K index of B (since B is col-major, first index = K). But n_idx is the N index in our weight tensor!

This is the bug! The B fragment's first index is K (for col-major), but we're using n_idx (the N index) as the first index!

The correct mapping:

b[0] = B[K_idx = lane_id/4][N_idx = (lane_id%4)*4..+4]
In col-major storage: smem_b[K_idx * 8 + N_start + i] for i=0..3
So the B fragment should read:


int k_idx = lane_id / 4;int n_start = (lane_id % 4) * 4;// b_regs[0] = B[k_idx][n_start..n_start+3] = 4 consecutive bytes at k_idx*8 + n_startmemcpy(&b_regs[0], &smem_b[k_idx * 8 + n_start], 4);// b_regs[1] = B[k_idx][n_start+16..n_start+19] = 4 consecutive bytes at k_idx*8 + n_start + 16memcpy(&b_regs[1], &smem_b[k_idx * 8 + n_start + 16], 4);
In our kernel, the B fragment reads:


int n_idx = wn + lane_id / 4;  // ← This should be K_idx, not N_idx!int k0 = sk + (lane_id % 4) * 4;  // ← This should be N_start, not K offset!
We have N and K swapped in the B fragment! The n_idx should be the K index within the tile, and k0/k1 should be the N offsets!

Let me fix the kernel:

Edited
w4a8_fp8_qmma.cu+22-9
Wait, I also need to check the A fragment. The A fragment is row-majow-major means A[row][col] at row * 32 + col. Thread t: a[0] = A[t/4][(t%4)*4..+4] = 4 consecutive bytes at row t/4, col (t%4)*4. In row-major storage, these are at positions (t/4)*32 + (t%4)*4 + i for i=0..3 — these ARE consecutive. ✓

So the A fragment is correct. Only the B fragment has the N/K swap. Let me also verify the B fragment read is correct:

For lane_id=0, wn=0, sk=0: k_idx=0, n_start=0. b_regs[0] reads W_fp8[0..3][0] = weights at (n=0..3, k=0). These are B[0][0..3] in col-major = B[K=0][N=0. ✓

For lane_id=1, wn=0, sk=0: k_idx=0, n_start=4. b_regs[0] reads W_fp8[4..7][0] = weights at (n=4..7, k=0). These are B[0][4..7] in col-major = B[K=0][N=4..7]. ✓

For lane_id=4, wn=0, sk=0: k_idx=1, n_start=0. b_regs[0] reads W_fp8[0..3][1] = weights at (n=0..3, k=1). These are B[1][0..3] in col-major = B[K=1][N=0..3]. ✓

This is correct! Now let me build and test:

Executed commands in the sglang directory
The b_regs declaration needs to be before the new code block:

Read w4a8_fp8_qmma.cu, lines  145

I see — the uint32_t b_regs[2]; is missing. It was removed when the old code was replaced. Let me add it:

Updated w4a8_fp8_qmma.cu and ran commands
The fix didn't help — now ALL cases fail, including the ones that previously passed. The B fragment read is still wrong. Let me reconsider.

The issue is that I'm gathering individual bytes from W_fp8[n_start + i][k_idx] but in col-major B, the 4 bytes in b_regs[0] should be B[k_idx][n_start..n_start+3]. In col-major storage, B[k][n] is at position k + n. So:

B[k_idx][n_start+0] → k_idx * 8 + n_start + 0
B[k_idx][n_start+1] → k_idx * 8 + n_start + 1
B[k_idx][n_start+2] → k_idx * 8 + n_start + 2
B[k_idx][n_start+3] → k_idx * 8 + n_start + 3
These ARE 4 consecutive bytes in col-major storage! So memcpy(&b_regs[0], &smem_b[k_idx * 8 + n_start], 4) is correct for col-major B.

But our SMEM stores weights as W_fp8[n][k] (row-major, [N][K]). So W_fp8[n][k] is at position n * kTileK + k. The value B[k][n] = W_fp8[n][k]. So:

B[k_idx][n_start+0] = W_fpt+0][k_idx] → (n_start+0) * kTileK + k_idx
B[k_idx][n_start+1] = W_fp8[n_start+1][k_idx] → (n_start+1) * kTileK + k_idx
B[k_idx][n_start+2] = W_fp8[n_start+2][k_idx] → (n_start+2) * kTileK + k_idx
B[k_idx][n_start+3] = W_fp8[n_start+3][k_idx] → (n_start+3) * kTileK + k_idx
These are NOT consecutive in SMEM (stride = kTileK = 64). So we need to gather them individually.

My gather code does exactly this:


b0_bytes[i] = (uint8_t)W_fp8[(n_start + i) * kTileK + k_idx];
This reads W_fp8[n_start+i][k_idx] which is B[k_idx][n_start+i]. ✓

But the result is worse than before. Let me check if the byte order in b_regs[0] matters. The MMA expects the uint32_t to contain the 4 FP8 values in a specific byte order. Let me check:

b0_bytes[0] = B[k_idx][n_start] → byte 0 (bits 0-7) of b_regs[0]
b0_bytes[1] = B[k_idx][n_start+1] → byte 1 (bits 8-15)
b0_bytes[2] = B[k_idx][n_start+2] → byte 2 (bits 16-23)
b0_bytes[3] = B[k_idx][n_start+3] → byte 3 (bits 24-31)

Wait — the PTX ISA says b[0] = B[t/44]. The second index is (t%4)*4..+4, which are 4 consecutive values along the N dimension (second index of col-major B). In col-major storage k*8+n, these are at positions (t/4)*8 + (t%4)*4 + i for i=0..3. These ARE consecutive. ✓

So memcpy(&b_regs[0], &smem_b[(t/4)*8 + (t%4)*4], 4) is correct for col-major B stored as B[k][n] at k*8+n.

But our weights are stored as W_fp8[n][k] at n * kTileK + k. For the MMA to read B[k][n], it needs to access W_fp8[n][k]. The 4 consecutive bytes in col-major B are at k+n to k*8+n+3. These correspond to W_fp8[n][k], W_fp8[n+1][k], W_fp8[n+2][k], W_fp8[n+3][k].

So the gather should read:


b0_bytes[0] = W_fp8[(n_start+0) * kTileK + k_idx]b0_bytes[1] = W_fp8[(n_start+1) * kTileK + k_idx]b0_bytes[2] = W_fp8[(n_start+2) * kTileK + k_idx]b0_bytes[3] = W_fp8[(n_start+3) * kTileK + k_idx]
This is exactly what my code does. But the result is wrong. Let me check if the issue is that n_start can exceed the SMEM bounds. For wn = ns * 8 where ns ranges 0..15,es 0..120. n_start = wn + (lane_id % 4) * 4 max = 120 + 3*4 = 132. n_start + 16 + 3 max = 132 + 19 = 151. W_fp8 has kTileN * kTileK = 128 * 64 = 8192 elements. W_fp8[151 * 64 + 31] = W_fp8[9695] which is out of bounds (max index 8191)!

That's the bug! n_start can exceed kTileN - 1 = 127! For wn = 120 and lane_id % 4 = 3: n_start = 120 + 12 = 132. W_fp8[132 * 64 + k_idx] is out of bounds!

The old code had the same issue but it happened to work because it only read W_fp8[n_idx * kTileK + k0] where n_idx = wn + lane_id / 4 max = 120 + 7 = 127. ✓

So the old code used n_idx = wn + lane_id / 4 (max 127) which was within bounds, but it was reading the wrong values (N and K swapped). My fix correctly reads the values but goes out of bounds!

The correct fix needs to ensure n_start + i < kTileN. Since wn can go up to 120 and (lane_id % 4) * 4 can go up to 12, n_start max = 132. But kTileN = 128. So we need to cap at 127.

But wait — the B fragment only needs 8 N rows. With wn = ns * 8 and ns from 0 to 15, wn go, 8, 16, ..., 120. For each wn, the B fragment reads N-rows wn to wn + 7 (since n_start + lane_id/4 max = wn + 7). But my code reads n_start + i where n_start = wn + (lane_id % 4) * 4. For lane_id % 4 = 3: n_start = wn + 12. Then n_start + 3 = wn + 15. But the B fragment only has 8 N-rows!

The issue is that n_start is the N start position within the B fragment, which has 8 N-rows. The 4 bytes in b_regs[0] should be at N positions n_start to n_start + 3 within the B fragment's 8 N-rows. The absolute N index is wn + n_start to wn + n_start + 3.

But n_start = (lane_id % 4) * 4 which ranges 0, 4, 8, 12. So the 4 bytes cover N positions wn + 0..3, wn + 4..7, wn + 8..11, wn + 12..15. But the B fragment only has 8 N-rows! So positions wn + 8..15 are outside the B fragment!

Wait — the B fragment is [32, 8] in col-major, meaning 32 K-values and 8 N-values. So n_start should only go up to 4 (covering N positions 0..3 and 4..7 within the 8-wide N tile). But (lane_id % 4) * 4 gives 0, 4, 8, 12. Values 8 and 12 are oside the 8-wide N tile!

This means my understanding of the B fragment layout is still wrong. Let me re-examine.

The PTX ISA says: b[0] = B[t/4][(t%4)*4..+4]. For t in 0..31:

t/4 ranges 0..7
(t%4)*4 ranges 0, 4, 8, 12
The first index t/4 ranges 0..7, which is the K dimension (32/4 = 8 groups of 4).
The second index (t%4)*4 ranges 0, 4, 8, 12, which are the N dimension start positions.

Wait — for b[0], the second index gives 4 consecutive values. If B is [32, 8] col-major, then the first index (K) range0..31 and the second index (N) ranges 0..7. But (t%4)*4 gives 0, 4, 8, 12 — and 8, 12 are OUTSIDE the N range 0..7!

This means B is NOT [32, 8] but rather [32, 16]! Or maybe b[0] reads 4 consecutive values along the K dimension, not the N dimension.

Let me reconsider. For col-major B, the inner dimension is K and the outer is N. So B[k][n] at k * stride_n + n. The 4 consecutive bytes would be 4 consecutive K values at the same N: B[k..k+3][n].

In col-major storage B[k][n] at k * 8 + n:

B[k][n] at posion k * 8 + n
B[k+1][n] at position (k+1) * 8 + n
B[k+2][n] at position (k+2) * 8 + n
B[k+3][n] at position (k+3) * 8 + n
These are NOT consecutive in memory (stride 8)!

But if the storage is B[k][n] at k + n * 32 (row-major with K as the inner dimension):

B[k][n] at position k + n * 32
B[k+1][n] at position k+1 + n * 32
These ARE consecutive!
So b_regs[0] reads 4 consecutive bytes which are 4 consecutive K values at the same N. In our SMEM W_fp8[n][k] (row-major [N][K]), W_fp8[n][k..k+3] are 4 consecutive K values at the same N. These ARE consecutive in SMEM!

So the original code was CORRECT:


memcpy(&b_regs[0], &W_fp8[n_idx * kTileK + k0], 4);
This reads 4 consecutive bytes at N-index n_idx, K-positions k0..k0+3. These are B[k0..k0+3][n_idx] in col-major. ✓

But wait — the PTX ISA says b[0] = B[t/4][(t%4)*4..+4]. For col-major B, the first index is K and the second is N. So b[0] = B[K=t/4][N=(t%4)*4..+4]. This means b[0] contains 4 consecutive N values at the same K.

But in memory, these are Ntive if B is stored as B[k][n] at k + n * 32 (K-inner). The 4 N values B[k][n..n+3] are at positions k + n*32, k + (n+1)*32, k + (n+2)*32, k + (n+3)*32 — stride 32!

But the PTX ISA says b[0] is 4 consecutive bytes. So either:

B is stored as B[k][n] at k*8 + n (N-inner), and b[0] reads 4 consecutive N values
B is stored as B[k][n] at k + n*32 (K-inner), and b[0] reads 4 consecutive K values
Let me check with the working test. In my test, B was stored as smem_b[k * 8 + n] (N-inner, col-major) and b_regs[0was read as memcpy(&b_regs[0], &smem_b[k0 * 8 + n_idx], 4) which reads 4 consecutive N values. And it produced correct results (all 64.0) for uniform B.

So the correct approach is:

Store B in SMEM as B[k][n] at k * 8 + n (N-inner, col-major)
Read b_regs[0] as 4 consecutive bytes from B[k_idx][n_start] = smem_b[k_idx * 8 + n_start]
In our kernel, we store W as W_fp8[n][k] at n * kTileK + k (K-inner, row-major). This is the TRANSPOSE of col-major B. So we need to either:
a. Store W in col-major format in SMEM, OR
b. Gather individual bytes from row-major W to form col-major B fragments

Option (a) is more efficient. Let me change the dequant to store weights in col-major format W_fp8[k * kTileN + n]:

Actually, the simplest fix is to change the SMEM storage format. Instead of W_fp8[n][k], store as W_fp8[k][n] (col-major). Then b_regs[0] reads 4 consecutive bytes from W_fp8[k_idx * kTileN + n_start].

Let me update the dequant to store in col-major format:

Edited
w4a8_fp8_qmma.cu+5-6
Now update the B fragment to read from col-major storage:

Edited
w4a8_fp8_qmma.cu+7-18
Now I also need to check: with col-major storage W_fp8[k * kTileN + n], the SMEM size is the same (kTileK * kTileN). But the access pattern in the B fragment needs k0 + 16 < kTileK. Since kTileK = 64 and k0 + 16 max = sk + 7 + 16 = 32 + 23 = 55 < 64. ✓

And n_start + 3 max = wn + 12 + 3 = 120 + 15 = 135. But kTileN = 128! So n_start + 3 can be 135 which is out of bounds!

Wait — the B fragment only covers 8 N-rows. wn = ns * 8 where ns range.15. But n_start = wn + (lane_id % 4) * 4 can go up to wn + 12. For wn = 120: n_start = 132. n_start + 3 = 135 > 127!

This is the same out-of-bounds issue. The B fragment reads W_fp8[k0 * kTileN + n_start + i] for i=0..3. If n_start + i >= kTileN, this reads out of bounds.

But wait — n_start is the N position within the B fragment's 8 N-rows. The B fragment has 8 N-rows, and (lane_id % 4) * 4 gives N offsets 0, 4, 8, 12. But only 0 and 4 are within the 8 N-rows! Offsets 8 and 12 are outside!

This meanshe B fragment layout must be different from what I assumed. Let me check: maybe the 4 bytes in b_regs[0] are NOT along the N dimension, but along the K dimension.

If B is col-major [32, 8] and stored as B[k][n] at k + n * 32 (K-inner), then:

b_regs[0] = 4 consecutive bytes at B[t/4][(t%4)*4..+4]
First index t/4 ranges 0..7 → K indices 0..7 (each group of 4 threads handles one K)
Second index (t%4)*4 ranges 0, 4, 8, 12 → N start positions
But N only has 8 values (0..7). With offsets 0, 4, 8, 12, offsetand 12 are invalid.

Unless the storage is B[k][n] at k * 8 + n (N-inner), and b_regs[0] reads 4 consecutive N values:

B[t/4][(t%4)*4..+4] = 4 consecutive N values at K-index t/4
In storage: (t/4)*8 + (t%4)*4 + i for i=0..3
For t%4 = 2: offset = 8, reads positions 8, 9, 10, 11 — but N only has 8 values!
This doesn't work either. Let me look at the actual PTX ISA documentation for the exact layout.

Running cd /home/oldzhu/sglang && python3 scripts/fcloud/f... in background — Show
cd /home/oldzhu/sglangpython3 scripts/fcloud/fcloud_exec.py exec --timeout 15 'cat > /tmp/check_ptx.cu << "XEOF"
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>

// Test: what does the MMA expect for the B fragment?
// We'll try different layouts and see which one works.
__global__ void test_b_layout(float* out, __nv_fp8_e4m3* B_col, int layout_mode) {
    int lane_id = threadIdx.x;
    __shared__ __nv_fp8_e4m3 smem_a[512];
    __shared__ __nv_fp8_e4m3 smem_b[512];  // enough space
    
    // A is all 1.0
    for (int i = threadIdx.x; i < 512; i += 32) smem_a[i] = __nv_fp8_e4m3(1.0f);
    
    // B: values = k*10 + n (distinct per position)
    // Load B_col (already in the test layout)
    for (int i = threadIdx.x; i < 256; i += 32) smem_b[i] = B_col[i];
    __syncthreads();
    
    uint32_t a_regs[4];
    int row0 = lane_id / 4, row1 = row0 + 8;
    int col0 = (lane_id % 4) * 4, col1 = col0 + 16;
    memcpy(&a_regs[0], &smem_a[row0 * 32 + col0], 4);
    memcpy(&a_regs[1], &smem_a[row0 * 32 + col1], 4);
    memcpy(&a_regs[2], &smem_a[row1 * 32 + col0], 4);
    memcpy(&a_regs[3], &smem_a[row1 * 32 + col1], 4);
    
    uint32_t b_regs[2];
    if (layout_mode == 0) {
        // Mode 0: B[k][n] at k*8+n, read 4 consecutive N values
        int k = lane_id / 4;
        int n = (lane_id % 4) * 4;
        memcpy(&b_regs[0], &smem_b[k * 8 + n], 4);
        memcpy(&b_regs[1], &smem_b[k * 8 + n + 16], 4);
    } else if (layout_mode == 1) {
        // Mode 1: B[k][n] at k + n*32, read 4 consecutive K values
        int n = lane_id / 4;
        int k = (lane_id % 4) * 4;
        memcpy(&b_regs[0], &smem_b[n * 32 + k], 4);
        memcpy(&b_regs[1], &smem_b[n * 32 + k + 16], 4);
    } else if (layout_mode == 2) {
        // Mode 2: B[k][n] at k*16 + n, read 4 consecutive N values (stride 16)
        int k = lane_id / 4;
        int n = (lane_id % 4) * 4;
        memcpy(&b_regs[0], &smem_b[k * 16 + n], 4);
        memcpy(&b_regs[1], &smem_b[k * 16 + n + 16], 4);
    }
    
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
          "r"(b_regs[0]), "r"(b_regs[1]),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    
    out[lane_id * 4 + 0] = c0;
    out[lane_id * 4 + 1] = c1;
    out[lane_id * 4 + 2] = c2;
    out[lane_id * 4 + 3] = c3;
}

int main() {
    __nv_fp8_e4m3 h_B[256];
    float *d_out;
    cudaMalloc(&d_out, 128 * 4);
    
    for (int mode = 0; mode < 3; mode++) {
        // Fill B with values = k*10 + n (distinct per (k,n))
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                h_B[n * 32 + k] = __nv_fp8_e4m3((float)(k * 10 + n));
        
        __nv_fp8_e4m3 *d_B;
        cudaMalloc(&d_B, 256);
        cudaMemcpy(d_B, h_B, 256, cudaMemcpyHostToDevice);
        
        test_b_layout<<<1, 32>>>(d_out, d_B, mode);
        cudaDeviceSynchronize();
        
        float h_out[128];
        cudaMemcpy(h_out, d_out, 128 * 4, cudaMemcpyDeviceToHost);
        
        // Check D[0][0]: should be sum_k A[0][k]*B[0][k] = sum_k (k*10+0) = 4960
        float d00 = h_out[0];  // thread 0: D[0][0]
        printf("Mode %d: D[0][0]=%f (expected 4960)\n", mode, d00);
        
        cudaFree(d_B);
    }
    cudaFree(d_out);
    return 0;
}
XEOF
nvcc -arch=sm_120a -o /tmp/check_ptx /tmp/check_ptx.cu 2>&1 && /tmp/check_ptx' 2>&1

bash: syntax error near unexpected token `&'
bash: syntax error near unexpected token `}'
float: command not found
bash: syntax error near unexpected token `('
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 : command not found
{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n: command not found
bash: syntax error near unexpected token `('
bash: syntax error near unexpected token `('
bash: syntax error near unexpected token `b_regs[0]'
bash: syntax error near unexpected token `c0'
The heredoc is being parsed by bash instead of being sent as-is. The issue is with the quoting. Let me write the test file locally and upload it:

Created test_b_layout.cu and ran Python scripts
All three modes give the same wrong answer (4936 instead of 4960). The small difference (24) might be due to FP8 quantization error. Let me check: with B[k][n] = k10 + n, the max value is 3110+7 = 317. In FP8 e4m3, 317 rounds to... let me check:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
The sum is 4936, not 4960, due to FP8 quantization error. So all three modes give the correct result (within FP8 precision). The MMA instruction works correctly with all three layouts!

This means the issue in our kernel is NOT in the B fragment layout. Let me go back to the original code and check what's different between our kernel and the working test.

The key difference: our working test uses A = all 1.0, while our failing test uses A = randn. But wait — the A fragment was verified correct. And the B fragment was verified correct with e original layout.

Let me revert to the original B fragment code and re-test:

Updated w4a8_fp8_qmma.cu
Now build and test with the original B fragment code but the "=f" + store MMA pattern:

Executed commands in the terminal
Back to the same pattern: Cases 1 and 3 pass (ones activations), Cases 2 and 4 fail (random activations).

Now I know the B fragment is correct (proven by the working test). The A fragment is correct (verified by diagnostic). The MMA instruction works. So what's different between our kernel and the working test?

Let me compare the kernel's MMA loop structure with the working test:

Working test:

1 warp, 1 MMA call
A loaded from SMEM with stride 32
B loaded from SMEM with stride 8 (col-major)
1 MMA call
Our kernel:

4 warps
Multiple MMA calls per warp (ms=0,1; ns=0..15; sk=0,32)
A loaded from SMEM with stride kTileK=64
B loaded from SMEM with stride kTileK=64
Multiple MMA calls with accumulation
The key difference: our kernel uses stride 64 for both A and B SMEM access, while the working test uses stride 32 for A and stride 8 for B.

Wait — that's because our SMEM has kTileK = 64 elements per row, while the MMA operates on 32 K-elements per call. The remaining 32 K-elements are in the next sk iteration. So the stride is 64, but each MMA call only uses 32 of them. This is correct because:

For sk=0: reads A[row][0..31] and B[row][0..31]
For sk=32: reads A[row][32..63] and B[row][32..63]
But wait — the A fragment reads A_fp8_smem[row0 * kTileK + col0] where kTileK = 64. For sk=0, col0=eads A_fp8_smem[row0 * 64 + 0]. For sk=32, col0=32: reads A_fp8_smem[row0 * 64 + 32]. ✓

For the B fragment: W_fp8[n_idx * kTileK + k0] where kTileK = 64. For sk=0, k0=0: reads W_fp8[n_idx * 64 + 0]. For sk=32, k0=32: reads W_fp8[n_idx * 64 + 32]. ✓

So the stride is correct. But the MMA expects B in col-major with stride 8 (for N=8). Our SMEM has stride 64. The B fragment reads 4 consecutive bytes which are 4 consecutive K values at the same N. In our SMEM W_fp8[n][k] at n * 64 + k, these are at positin*64 + k, n*64 + k+1, n*64 + k+2, n*64 + k+3. These are indeed 4 consecutive K values. ✓

OK, I'm going to try a completely different approach. Let me check if the issue is with the __nv_fp8_e4m3 type used in SMEM. Maybe the MMA instruction expects a different FP8 type (like __nv_fp8_e4m3fn instead of __nv_fp8_e4m3):

Running cd /home/oldzhu/sglang && python3 scripts/fcloud/f... in background — Show
cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 15 'cat > /tmp/check_fp8_tcu << "XEOF"
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
int main() {
    printf("__nv_fp8_e4m3  size=%zu\n", sizeof(__nv_fp8_e4m3));
    printf("__nv_fp8_e4m3fn size=%zu\n", sizeof(__nv_fp8_e4m3fn));
    // Check if they're the same type
    __nv_fp8_e4m3 v1 = __nv_fp8_e4m3(1.0f);
    __nv_fp8_e4m3fn v2 = __nv_fp8_e4m3fn(1.0f);
    uint8_t b1, b2;
    memcpy(&b1, &v1, 1);
    memcpy(&b2, &v2, 1);
    printf("e4m3(1.0) = 0x%02x, e4m3fn(1.0) = 0x%02x\n", b1, b2);
    return 0;
}
XEOF
nvcc -arch=sm_120a -o /tmp/check_fp8_type /tmp/check_fp8_type.cu && /tmp/check_fp8_type' 2>&1

bash: syntax error near unexpected token `('
uint8_t: command not found
bash: syntax error near unexpected token `&'
bash: syntax error near unexpected token `&'
bash: syntax error near unexpected token `"e4m3(1.0) = 0x%02x, e4m3fn(1.0) = 0x%
02x\n",'
bash: return: can only `return' from a function or sourced script
bash: syntax error near unexpected token `}'
XEOF: command not found
>  
The heredoc issue continues. Let me stop fighting with the SSH heredoc and instead focus on what I know. The instance is running and costing money. Let me pause it and analyze the problem offline.

Executed command to pause workflow in fcloud script
Instance paused. Now let me summarize the current status and what I've learned.

Current status: The kernel passes the synthetic uniform test (all weights=7, all activations=1.0) but fails with non-uniform activations. The MMA instruction works correctly on SM120. The A and B fragment values are correct. The bug is still unidentified.

Key findings:

✅ A fragment values verified correct (diagnostic printf)
✅ B fragment layout correct (proven by working MMA test)
✅ MMA instruction works on SM120 (verified with standalone test)
✅ __nv_fp8_e4m3 constructor works correctly
❌ Kernel fails with non-uniform activations (Cases 2, 4)
✅ Kernel passes with uniform activations (Cases 1, 3)
Remaining possibilities:

The "=f" + store pattern for MMA output might have a subtle issue with the compiler way the store
The SMEM might have alignment issues
The multiple MMA calls across sk iterations might have a bug in the accumulation
There might be a race condition between warps
Let me check possibility #1 — the "=f" + store pattern. The compiler might optimize away the store to cp if it thinks cp is dead after the asm block. Let me use "+f" instead:

Edited
w4a8_fp8_qmma.cu+2-4
Wait, this is the same as the original code before my change. The original code used "+f" and it still failed. Let me check the tual current state of the file:

Read w4a8_fp8_qmma.cu, lines 115 to 155

Good — the code is back to the original "+f" pattern (which was the original before any of my changes). The issue is not in the MMA inline asm pattern.

Let me now look at this problem from a completely different angle. Let me check if the issue is with the epilogue. The epilogue writes c_regs[ms][ns] to the output. But c_regs is indexed as c_regs[ms][ns] where ms is 0 or 1 (for the two 16-row halves of the warp) and ns is 0..15 (fothe 16 N-tiles).

Wait — the MMA loop iterates ms and ns:


for (int ms = 0; ms < 2; ++ms) {    int wm = warp_m0 + ms * kMmaM;    for (int ns = 0; ns < 16; ++ns) {        int wn = ns * kMmaN;        ...        float* cp = c_regs[ms][ns];        ...        asm("mma.sync ..." : "+f"(cp[0]), ...);    }}
The c_regs[ms][ns] array is float[2][16][4]. Each MMA call uses a different c_regs[ms][ns] slice. This is correct — epair corresponds to a different output tile.

But wait — the sk loop is OUTSIDE the ms and ns loops:


for (int sk = 0; sk < kTileK; sk += kMmaK) {    for (int ms = 0; ms < 2; ++ms) {        for (int ns = 0; ns < 16; ++ns) {            ...            asm("mma.sync ..." : "+f"(cp[0]), ...);        }    }}
For sk=0, the MMA accumulates into c_regs[ms][ns]. For sk=32, the MMA accumulates again into the same c_regs[ms][ns]. This correct — the two K-tile halves are accumulated.

But the A and B fragments for sk=0 read from SMEM positions col0 = 0..12 and col1 = 16..28. For sk=32, they read from col0 = 32..44 and col1 = 48..60. These are different positions in SMEM, which were loaded in Phase 1/2. ✓

OK wait, I just realized something. The A fragment reads A_fp8_smem[row0 * kTileK + col0] where kTileK = 64. For sk=0, col0=0: reads A_fp8_smem[row0 * 64 + 0] which is the activation at global K position kb + 0. For sk=32, col0=32: r A_fp8_smem[row0 * 64 + 32] which is the activation at global K position kb + 32. ✓

This is all correct. I'm going to try one more thing — check if the issue is that the MMA accumulates into the same c_regs across the kb outer loop. The kb loop goes kb = 0, 64 for K=128. The c_regs are initialized to 0 before the kb loop and accumulate across both kb iterations. This should be correct.

Actually, let me check something I haven't verified — the output tensor stride. The kernel writes to c_bf16[(m0 + r ldc + n0 + col0]. The ldc parameter is c_bf16.stride(0). For a contiguous tensor, stride(0) = N. Let me verify:

Read w4a8_fp8_qmma.cu, lines 155 to 196

Wait — I notice something! The a_fp8 tensor is cast to const __nv_fp8_e4m3*, but the tensor's dtype is torch::kFloat8_e4m3fn (checked by TORCH_CHECK). The __nv_fp8_e4m3 and __nv_fp8_e4m3fn types might have different underlying representations!

Let me check: __nv_fp8_e4m3 is the type with NaN support, while __nv_fp8_e4m3fn is the finite-only variant. Th might have different bit patterns for the same float value!

If torch.float8_e4m3fn uses __nv_fp8_e4m3fn (finite-only) but our kernel casts to __nv_fp8_e4m3 (with NaN), the bit patterns might be different!

Let me verify:

Running cd /home/oldzhu/sglang && python3 scripts/fcloud/f... in background — Show
cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 30 'python3 -c "
import torch
# Check what FP8 type PyTorch uses
print(f'Torch FP8 dtype: {torch.float8_e4m3fn}')
t = torch.nsor([1.0, 2.0, -1.0, 0.5], dtype=torch.float8_e4m3fn)
print(f'Values: {t}')
print(f'Bytes: {t.view(torch.uint8).tolist()}')
# Compare with expected FP8 e4m3 values
# 1.0 = 0x38, 2.0 = 0x40, -1.0 = 0xb8, 0.5 = 0x30
print(f'Expected: [0x38, 0x40, 0xb8, 0x30]')
" 2>&1' 2>&1
The f prefix in the f-string is causing issues with the SSH command parsing. Let me write a script file:

Created test_fp8_dtype.py and ran command
The instance is paused (503). That's expected. Let me think about the FP8 type issue.

Actually, I just realized something critical. Let me look at the TORCH_CHECK in the C++ wrapper:


TORCH_CHECK(a_fp8.dtype() == torch::kFloat8_e4m3fn, "act must be fp8_e4m3");
It checks for torch::kFloat8_e4m3fn. But then it casts to const __nv_fp8_e4m3*:


static_cast<const __nv_fp8_e4m3*>(a_fp8.const_data_ptr())
If __nv_fp8_e4m3 and __nv_fp8_e4m3fn have different representations, this cast would cause the kernel to read wrong values from the activation tensor!

But wait — the diagnostic confirmed tharagment values match the expected values. So the cast is correct (they have the same representation).

Actually, I just realized I should check the __nv_fp8_e4m3 type more carefully. The __nv_fp8_e4m3 type might be the same as __nv_fp8_e4m3fn in terms of bit representation — both use IEEE 754 FP8 e4m3 format. The difference is only in how NaN/inf are handled (e4m3fn has no NaN, e4m3 has NaN). The bit patterns for normal values should be identical.

So this is not the issue.

Let me take a step back and thk about what could cause the kernel to work with uniform activations but fail with random activations. The only difference is the values in A_fp8_smem. If the A fragment reads correct values (verified), and the B fragment reads correct values (verified), and the MMA instruction works correctly (verified), then the only remaining possibility is...

The MMA instruction produces wrong results when the A fragment contains specific FP8 bit patterns!

This could happen if:

The MMA instruction has a bug for certain FP8 values (unlikely but possible on new hardware)
The FP8 values produced by __nv_fp8_e4m3(float) are not what the MMA expects
There's a NaN or inf in the FP8 values that causes the MMA to produce wrong results
Let me check possibility #3. The __nv_fp8_e4m3 constructor with __NV_SATFINITE (from __nv_cvt_float_to_fp8) saturates to finite values. But in our kernel, we use __nv_fp8_e4m3(fv) which is the direct constructor. Does this constructor saturate or produce NaN for out-of-range values?

From the earlier test: __nv_fp8_e4m3(1000.0f) gives 0x7e which is 448.0 (the max representable value). So it saturates. ✓

But what about negative values? __nv_fp8_e4m3(-1000.0f) should give 0xfe = -448.0. Let me check if the MMA handles negative FP8 values correctly.

Actually, the issue might be simpler. Let me check if the __nv_fp8_e4m3 constructor produces the NaN bit pattern for certain inputs. The FP8 e4m3 format has:

Max normal: 448.0 (0x7e for positive, 0xfe for negative)
NaN: 0x7f (all exponent bits 1, mantsa non-zero)
If __nv_fp8_e4m3(fv) produces 0x7f (NaN) for some float value, the MMA might produce wrong results.

The __nv_cvt_float_to_fp8 function with __NV_SATFINITE avoids NaN. But __nv_fp8_e4m3(fv) (the direct constructor) might not!

Actually, looking at the CUDA documentation: __nv_fp8_e4m3 is a type that can represent NaN (0x7f). The constructor might produce NaN for certain inputs. And the MMA instruction with e4m3 type (which supports NaN) would handle NaN values differently than finite values.

But our activations are randn values converted to float8_e4m3fn (finite-only). The weights are dequantized INT4 values. Neither should produce NaN.

OK, I'm going to look at this from a completely different angle. Let me check if the issue is with the __syncthreads() placement. The kernel has:


// Phase 1: dequant// Phase 2: load activations__syncthreads();// Phase 3: MMAfor (int sk = ...) {    for (int ms = ...) {        for (int ns = ...) {            // rd from SMEM, do MMA        }    }}__syncthreads();
There's only ONE __syncthreads() between Phase 2 and Phase 3. The MMA loop reads from SMEM (which was written in Phase 1 and 2). After the __syncthreads(), all threads see the SMEM contents. Then each thread reads from SMEM in the MMA loop. Since the MMA loop only reads (doesn't write to SMEM), no additional __syncthreads() is needed. ✓

But wait — there's another __syncthreads() at the end of the MMA loop (after the sk loop). This is ne before the next kb iteration (which writes to SMEM again). ✓

The __syncthreads() placement looks correct.

I'm truly stuck. Let me try the most direct debugging approach possible — add a diagnostic that prints the accumulator values AFTER the MMA completes, for both the sk=0 and sk=32 iterations:

Updated w4a8_fp8_qmma.cu and ran commands
Interesting! The diagnostic shows:

First 8 lines: Cases 1 and 3 (ones activations) — values look reasonable
Next 8 lines: Cases 2 and 4 (random activations) — vok reasonable
Last 4 lines: ???
Wait — the test has 4 cases, each running the kernel. The diagnostic fires for warp_id==0 && ms==0 && ns==0 && sk==0 && kb==0 && tid<4. Each case produces 4 lines (for tid=0..3). So we should see 16 lines total (4 cases × 4 threads). But there are 20 lines!

The extra 4 lines (last group) are from... let me check. Actually, the ms loop runs twice (ms=0 and ms=1), so the diagnostic fires for both ms=0 and ms=1. That's 8 lines per case. For 4 cases, that's 32 lines. But only are shown.

Wait — the diagnostic also fires for the second sk iteration (sk=32)? No, the condition checks sk == 0.

Actually, looking more carefully: the first 8 lines have large values (like -32, 0, 32, 64) — these are from Cases 1 and 3 (ones activations, correct). The next 8 lines have smaller values (like -1.29, 0, 1.29, 2.59) — these are from Cases 2 and 4 (random activations). The last 4 lines have values around 160-232 — these seem to be from a different case.

But the key observation is: tulator values after the first MMA call look reasonable for all cases. The issue must be in the subsequent MMA calls (sk=32, or the second kb iteration).

Let me add a diagnostic for the second sk iteration:

Updated w4a8_fp8_qmma.cu and ran commands
Now I can see the accumulator values for both sk=0 and sk=32. Let me verify the sk=32 values for Case 1 (ones activations, simple weights).

For Case 1 (first 8 lines, sk=0 and sk=32):

sk=0: tid=0: c={-32, 0, -32, 0}
sk=32: tid=0: c={-64, 0, -64, 0}
Total: c={-96, 0, -96, 0}

For the expected value: with a=ones(128, 128) and simple weights (k*N+n)%16 - 1:

D[0][0] = sum_k A[0][k] * W[k][0] = sum_k 1.0 * ((k*256+0)%16 - 1) = sum_k ((8k)%16 - 1)
For k=0..63: (8k)%16 = 0 for even k, 8 for odd k → values: -1, 7, -1, 7, ... → sum of 32 pairs of (-1+7) = 32*6 = 192
For k=64..127: same pattern → another 192
Total = 384
But the kernel gives c={-96, 0, -96, 0} which is completely wrong!

Wait — the diagnostic only shows the first MMA call (ms=0, ns=0). But there atiles (ns=0..15) and 2 M-halves (ms=0,1). The first MMA call (ms=0, ns=0) only covers D[0..15][0..7]. The total D[0][0] = sum over all MMA calls that contribute to D[0][0].

But for ms=0, ns=0, the MMA covers D[wm..wm+15][wn..wn+7] = D[0..15][0..7]. The accumulator for D[0][0] after sk=0 and sk=32 should be the partial sum over K=0..63. The total should be 192 (half of 384).

But the diagnostic shows c={-32, 0, -32, 0} for sk=0 and c={-64, 0, -64, 0} for sk=32. Total = {-96, 0, -96, 0}. But expected is 192!

Wait — the D fragment layout: thread t has d[0]=D[t/4][2*(t%4)], d[1]=D[t/4][2*(t%4)+1], d[2]=D[t/4+8][2*(t%4)], d[3]=D[t/4+8][2*(t%4)+1].

For tid=0: d[0]=D[0][0], d[1]=D[0][1], d[2]=D[8][0], d[3]=D[8][1].

So c[0] = D[0][0] = -96 after both sk iterations. But expected is 192!

This means the MMA is computing wrong values even for Case 1 (ones activations, simple weights)!

Wait — but Case 1 PASSES (maxdiff=0)! How can the accumulator be wrong but the output be correct?

Oh — I see! The diagnostic the accumulator for the first MMA call only (ms=0, ns=0). But there are 16 N-tiles (ns=0..15) and 2 M-halves (ms=0,1). The total output D[0][0] is the sum of contributions from all MMA calls that cover D[0][0].

But D[0][0] is only covered by one MMA call: ms=0, ns=0 (since D[0][0] is in the first M-half and first N-tile). So the accumulator after all MMA calls should be the final value for D[0][0].

But the diagnostic shows -96 while the expected is 384. And the test says PASS with maxdiff=0!

This means the diagnostic is wrong, or the output is wrong and the test is also wrong somehow.

Actually — wait. The diagnostic prints the accumulator AFTER the MMA instruction but BEFORE the sk loop completes for the second iteration. Let me check: the diagnostic is inside the sk loop, so it fires for both sk=0 and sk=32. The value for sk=32 is the accumulator AFTER the second MMA call.

For Case 1, tid=0: sk=0: c={-32, 0, -32, 0}, sk=32: c={-64, 0, -64, 0}. The total is {-96, 0, -96, 0}.

But the expected D[0][0] =84. And the test says PASS. How?

Let me check: the test_isolate.py Case 1 uses a = ones(M, K) and simple weights (k*N+n)%16. Let me verify the expected output:


w_ref[k, n] = float((k*N+n)%16) - 1.0
For n=0: w_ref[k, 0] = (k*256+0)%16 - 1 = (8k)%16 - 1.

k=0: 0-1 = -1
k=1: 8-1 = 7
k=2: 0-1 = -1
k=3: 8-1 = 7
...
Sum over k=0..127: 64*(-1) + 64*7 = 384. ✓

But the MMA gives -96. That's wrong!

But the test says PASS. So either:

The diagnostic is wrong (unlikely — it uses printf)
The test is wrong  — it uses a simple reference)
The diagnostic fires for a different case than what I think
Actually — the diagnostic might be firing for ms=0, ns=0 which covers D[0..15][0..7]. But the accumulator is for ALL MMA calls that contribute to these positions. The sk loop covers K=0..31 and K=32..63. But there's also the kb loop!

For K=128, kTileK=64, the kb loop goes kb=0 and kb=64. For each kb, the sk loop goes sk=0 and sk=32. So there are 4 MMA accumulations per output position!

The diagnostic only shows es for kb=0. The kb=64 values are not shown! So the total accumulator after both kb iterations would be:

kb=0, sk=0: {-32, 0, -32, 0}
kb=0, sk=32: {-64, 0, -64, 0} (cumulative)
kb=64, sk=0: ??? (not shown)
kb=64, sk=32: ??? (not shown)
The final value should be 384. If kb=0 gives -96, then kb=64 should contribute 480 to reach 384 total. That seems unlikely.

Unless the diagnostic fires for BOTH kb=0 and kb=64 but the output doesn't show it because the output is truncated. Let me check with grep -c:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
16 lines for tid=0. With 4 test cases and 2 ms values (ms=0,1), that's 4×2 = 8 firings per test case. But the diagnostic only fires for ms=0, ns=0. So 4 test cases × 1 ms × 1 ns = 4 firings per kb. With 2 kb iterations, that's 8. But we have 16 — double!

Oh — the diagnostic fires for both ms=0 and ms=1 (since warp_id==0 and ms is 0 or 1). So 4 test cases × 2 ms × 2 sk × 1 ns = 16. ✓

But the diagnostic doesn't fire for kb=64 because the conditb == 0! Wait, no — I removed the kb == 0 condition. Let me check:


if (warp_id == 0 && ms == 0 && ns == 0 && kb == 0 && tid < 4) {
Oh — kb == 0 is still there! The diagnostic only fires for kb=0. So we only see half the accumulations.

For K=128, kTileK=64, the kb loop has 2 iterations. The diagnostic only shows kb=0. The accumulator after kb=0 is -96 (for D[0][0]). After kb=64 (not shown), it should reach 384.

But -96 + x = 384 → x = 480. That means kb=64 contributes 480. But thfor kb=64 (k=64..127) have the same pattern as kb=0 (k=0..63). So the contribution should be the same: -96. But -96 + (-96) = -192, not 384!

Something is very wrong. Let me add kb to the diagnostic:

Updated w4a8_fp8_qmma.cu and ran terminal commands
Now I can see the full picture for Case 1 (first 4 groups × 4 threads = 16 lines, then repeated for ms=1):

For tid=0, D[0][0]:

kb=0, sk=0: -32
kb=0, sk=32: -64 (cumulative)
kb=64, sk=0: -96 (cumulative)
kb=64, sk=32: -128 (cumulative)
Final D[0][0] = -128. ut expected is 384!

But the test says PASS! How can -128 == 384?

Wait — the test uses torch.allclose(r, ref, atol=1.0). If r[0,0] = -128 and ref[0,0] = 384, the diff would be 512, which is >> 1.0. The test should FAIL!

Unless the test is wrong somehow. Let me check the test script:

Read test_isolate.py, lines 1 to 30

The test uses a_ones = torch.ones(M, K) but the kernel expects a_fp8 which is float8_e4m3fn. torch.ones(M, K) creates a float32 tensor, then .to(torch.float8_e4m3fn) converts it. All vals should be 1.0 = 0x38 in FP8.

But the diagnostic shows c={-32, 0, -32, 0} for the first MMA call with ones activations. Let me verify: with a=1.0 and w_ref[k,n] = (k*N+n)%16 - 1:

For D[0][0]: sum over k=0..31 of 1.0 * ((k*256+0)%16 - 1) = sum over k=0..31 of ((8k)%16 - 1)

For k=0: 0-1 = -1
For k=1: 8-1 = 7
For k=2: 0-1 = -1
For k=3: 8-1 = 7
...

Sum over k=0..31: 16*(-1) + 167 = 166 = 96

But the diagnostic shows -32 for D[0][0] after sk=0 (k=0..31). Not 96!

Wait — the diagnostic shows the accumulatofor the FIRST ns value (ns=0). But D[0][0] is in the first N-tile (ns=0, wn=0). So the first MMA call (ms=0, ns=0, sk=0) computes D[0..15][0..7] using A[0..15][0..31] and B[0..7][0..31].

For D[0][0]: sum over k=0..31 of A[0][k] * B[0][k] = sum over k=0..31 of 1.0 * W_fp8[0][k]

W_fp8[0][k] = dequantized weight at (n=0, k) = ((k*256+0)%16 - 1) = ((8k)%16 - 1)

For k=0: -1
k=1: 7
k=2: -1
k=3: 7
...

Sum over k=0..31: 16*(-1) + 16*7 = 96

But the MMA gives -32. That's wrong!

But the test says PASS! Let me check what ref[0,0] actually is:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Both kernel and reference give -128.0! So the kernel IS correct for Case 1! My manual calculation of 384 was wrong.

Let me recalculate: w_ref[k, n] = (k*N+n)%16 - 1. For n=0: (k*256)%16 - 1 = 0 - 1 = -1 for all k (since 256 is a multiple of 16)! So w_ref[k, 0] = -1 for all k!

Sum over k=0..127 of 1.0 * (-1) = -128. ✓

So Case 1 passes because the kernel and reference agree. The MMA is computing correctly!

Now let  check Case 2 (simple weights, random activations):

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Now I can see the problem clearly! For Case 2:

Kernel[0,0] = -4.25
Ref[0,0] = 7.6875
Diff = 11.9375
The kernel gives -4.25 but reference gives 7.6875. And the diagnostic shows the accumulator for tid=0 after all iterations is -4.25 (kb=64, sk=32: c[0]=-4.25). This matches the kernel output.

But the reference gives 7.6875. Let me verify: with a = randn and w_ref[k,n] = (k*N+n)%16 - 1:

For n=0: w_ref[k,0] = (k*256)%16 - 1 = -1 (since 256 is a multiple of 16).

So D[0][0] = sum_k a[0][k] * (-1) = -sum_k a[0][k].

Let me check what sum_k a[0][k] is:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
So sum a[0,:] = -7.6875, and D[0,0] = -sum = 7.6875. The reference gives 7.6875. But the kernel gives -4.25!

The kernel computes the wrong result. And the diagnostic shows the accumulator progressing:

kb=0, sk=0: -1.29
kb=0, sk=32: -3.38
kb=64, sk=0: -9.61
kb=64, sk=32: -4.25
The final value is -4.25, but expected is 7.6875. The accumulator is going in the wrong direction!

Wait — for kb=64, sk=0, the accumulator goes from -3.38 to -9.61 (decreases by 6.23). For kb=64, sk=32, it goes from -9.61 to -4.25 (increases by 5.36).

But for the reference, the contribution from kb=64 should be sum_{k=64}^{127} a[0][k] * (-1) = -sum_{k=64}^{127} a[0][k].

Let me check: sum a[0, 64:128]:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
So:

kb=0 (k=0..63): sum a[0,0:64] = -2.09, contribution to D[0] = -(-2.09) = 2.09
kb=64 (k=64..127): sum a[0,64:128] = -5.60, contribution to D[0,0] = -(-5.60) = 5.60
Total D[0,0] = 2.09 + 5.60 = 7.69 ✓
But the kernel gives:

kb=0, sk=0: -1.29 (expected partial: ~1.05)
kb=0, sk=32: -3.38 (expected partial: ~2.09)
kb=64, sk=0: -9.61 (expected: ~4.64)
kb=64, sk=32: -4.25 (expected: ~7.69)
The kernel values are negative when they should be positive! The MMA is computing the wrong sign!

This means the MMA is computing D -= A * B instead of D += A * B! Or the A or B frment values have the wrong sign!

Wait — the sign is consistently wrong. For Case 1 (ones activations), the kernel gave -128 and the reference gave -128. Both were negative, so the sign was correct for Case 1.

But for Case 2, the kernel gives -4.25 and the reference gives 7.69. The sign is wrong.

What's different? In Case 1, all A values are 1.0 (positive). In Case 2, A values are random (some positive, some negative).

If the MMA computes D += A * B but the A fragment has the wrong sign for some valuesthe result would be wrong. But the A fragment was verified correct!

Unless... the issue is with how negative FP8 values are handled. Let me check: a[0,0] = 0.1875 (positive), a[0,2] = -0.171875 (negative). The FP8 representation of 0.1875 is 0x24 (sign=0), and -0.171875 is 0xa3 (sign=1). Both should be handled correctly by the MMA.

Wait — let me check if __nv_fp8_e4m3 handles negative values correctly:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
The FP8 values look correct. Negative valuehave sign=1, positive have sign=0.

Wait — I just noticed something in the diagnostic output. For Case 2, tid=0:

kb=0, sk=0: c={-1.29, 0, 6.30, 0}
kb=0, sk=32: c={-3.38, 0, 9.99, 0}
c[0] = D[0][0] after kb=0 = -3.38 (should be ~2.09)
c[2] = D[8][0] after kb=0 = 9.99

But c[0] and c[2] should be different because they correspond to different M-rows (row 0 and row 8). The activations for row 0 and row 8 are different.

Now, for kb=64, sk=0: c={-9.61, 0, 12.96, 0}. The contribution from kb=64 is:

D[0][0]: .61 - (-3.38) = -6.23 (should be ~5.60)
D[8][0]: 12.96 - 9.99 = 2.97
The sign of the kb=64 contribution is wrong for D[0][0] (-6.23 instead of +5.60).

Let me check: sum a[8, 64:128]:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
So for D[0][0] (row 0, col 0): expected = -(-2.09) + -(-5.60) = 2.09 + 5.60 = 7.69
For D[8][0] (row 8, col 0): expected = -(-4.52) + -(-2.35) = 4.52 + 2.35 = 6.87

But the kernel gives:

D[0][0] = -4.25 (expected 7.69)
D[8][0] = 18.81 (expected 6.87)
The kernel's D[0][0] is wrong both in sign and magnitude, while D[8][0] is wrong in magnitude but has the right sign.

This is very confusing. Let me check if the issue is with the kb=64 iteration specifically. For kb=64, the activation values should be a[mg][kg] where kg = 64 + k. These are loaded in Phase 2.

Wait — I just realized something. The SMEM for activations is A_fp8_smem[m * kTileK + k]. For kb=64, the activation at (mg, kg=64+k) is stored at A_fp8_smem[m * 64 + k]. The A fragment reads A_fp8_smem[row0 * 64 + c0] where col0 = sk + (lane_id % 4) * 4.

For sk=0: col0 = 0, 4, 8, 12. These read A_fp8_smem[row0*64 + 0], A_fp8_smem[row0*64 + 4], etc. These are the activations at K-positions kb + 0, kb + 4, kb + 8, kb + 12 = 64, 68, 72, 76. ✓

For sk=32: col0 = 32, 36, 40, 44. These read activations at K-positions 64+32=96, 100, 104, 108. ✓

This is correct. Let me check if the A fragment for kb=64 reads the right values. The diagnostic only shows kb=0 for the A fragment (the old diagnostic was removed). But the accator values for kb=64 are suspicious.

Actually, let me check something simpler. The diagnostic shows that for Case 2, the accumulator for D[0][0] goes:
-3.38 → -9.61 → -4.25

The jump from -9.61 to -4.25 (an increase of 5.36) is for kb=64, sk=32. This should be the contribution from K=96..127. Let me check what sum a[0, 96:128] is:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
So for D[0][0] = -sum(a[0,:]) = 7.69:

kb=0, sk=0 (k=0..31): contribution = -(-2.54) = 2.54
kb=0, sk=32 (k=32..63)ntribution = -(0.45) = -0.45
kb=64, sk=0 (k=64..95): contribution = -(-1.19) = 1.19
kb=64, sk=32 (k=96..127): contribution = -(-4.41) = 4.41
Expected accumulator progression:

After kb=0, sk=0: 2.54
After kb=0, sk=32: 2.54 - 0.45 = 2.09
After kb=64, sk=0: 2.09 + 1.19 = 3.28
After kb=64, sk=32: 3.28 + 4.41 = 7.69
Kernel accumulator progression:

After kb=0, sk=0: -1.29
After kb=0, sk=32: -3.38
After kb=64, sk=0: -9.61
After kb=64, sk=32: -4.25
The kernel values are negative when they should be positive, and the magnitudes are different. This is NOT a simple sign flip — the magnitudes are also wrong.

Wait — the kernel's first value is -1.29, but the expected is 2.54. The ratio is approximately -0.5. Let me check if the kernel is reading the wrong activation values.

Actually, I just had a breakthrough insight. Let me look at the sk loop more carefully:


for (int sk = 0; sk < kTileK; sk += kMmaK) {
kTileK = 64, kMmaK = 32. So sk goes 0, 32. For each sk, the A fragment reads:

col0 = sk + (lane_i = col0 + 16
For sk=0: col0 = 0, 4, 8, 12, col1 = 16, 20, 24, 28. Total K range: 0..31. ✓
For sk=32: col0 = 32, 36, 40, 44, col1 = 48, 52, 56, 60. Total K range: 32..63. ✓

But wait — the A_fp8_smem has kTileK = 64 columns. So the A fragment reads K-positions 0..31 for sk=0 and 32..63 for sk=32. These are the correct positions within the 64-wide K tile.

But the B fragment also reads:

k0 = sk + (lane_id % 4) * 4
k1 = k0 + 16
For sk=0: k0 = 0, 4, 8, 12, k1 = 16, 20, 24, 28. K range: 0..31. ✓
For sk= 32, 36, 40, 44, k1 = 48, 52, 56, 60. K range: 32..63. ✓

Both A and B read from the same K positions. The MMA computes D += A * B for each K slice. This should be correct.

But the accumulator values are wrong! Let me check if the issue is with the sk loop bounds. sk < kTileK = 64, step 32. So sk = 0, 32. After sk = 32, the next would be sk = 64 which is not < 64. ✓

I'm going to try one more thing — check if the issue is that the MMA instruction on SM120 has a different K stride than expected. Maybe120, m16n8k32 actually uses K=16 or K=64 instead of K=32:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Both tests give values close to expected (within FP8 quantization error). So the MMA works correctly with varied A values!

This means the issue in our kernel is NOT in the MMA instruction itself. The problem must be in how our kernel loads data into SMEM or iterates over tiles.

Let me now compare our kernel's A fragment read with the working test's A fragment read:

Working test: smem_a[row0 * 32 + col0] — stride 32, A tile is [16, 32]
Our kernel: A_fp8_smem[row0 * kTileK + col0] — stride 64, A tile is [16, 64]

The working test uses stride 32 because the A tile is exactly 32 columns wide. Our kernel uses stride 64 because the SMEM has kTileK = 64 columns, but the MMA only operates on 32 K-columns at a time.

This is the key difference! In the working test, the A fragment reads from a contiguous [16, 32] tile. In our kernel, the A fragment reads from a [16, 64] SMEM with stride 64, but onhe first or second 32 columns are used per sk iteration.

For sk=0: reads from columns 0..31 of the [16, 64] SMEM. ✓
For sk=32: reads from columns 32..63 of the [16, 64] SMEM. ✓

But wait — the A fragment for sk=0 reads col0 = 0, 4, 8, 12 and col1 = 16, 20, 24, 28. These are columns 0..31. For sk=32: col0 = 32, 36, 40, 44 and col1 = 48, 52, 56, 60. These are columns 32..63. ✓

This is correct. But let me verify by checking what values the A fragment reads for sk=32:

Actually, I just realized I shou if the issue is with the B fragment in our kernel. In the working test, B is stored as col-major B[k][n] at k * 8 + n. In our kernel, B is stored as row-major W_fp8[n][k] at n * 64 + k.

The B fragment in the working test reads smem_b[k0 * 8 + n_idx] which is 4 consecutive bytes at K-index k0, N-indices n_idx..n_idx+3. In col-major, these are 4 consecutive N values at the same K.

In our kernel, the B fragment reads W_fp8[n_idx * 64 + k0] which is 4 consecutive bytes at N-index n_idx, K-indices k0..k0+3. In row-major, these are 4 consecutive K values at the same N.

The MMA expects B in col-major format. In col-major, b[0] should be 4 consecutive values at the same K (first index), spanning N (second index). But our kernel reads 4 consecutive values at the same N, spanning K!

This is the bug! The B fragment reads K-consecutive values (row-major style) but the MMA expects N-consecutive values (col-major style).

Wait — but the working test with uniform B values passed. And my standalone test with varied B lues also passed (D[0][0] = 4538 ≈ 4936). Let me check: in the standalone test, the B fragment reads smem_b[k0 * 8 + n_idx] which is 4 consecutive N values at the same K. This is the col-major layout. ✓

In our kernel, the B fragment reads W_fp8[n_idx * 64 + k0] which is 4 consecutive K values at the same N. This is the row-major layout. But the MMA expects col-major!

So the B fragment in our kernel reads the wrong values! But it happens to work when all B values are the same (uniform weights) or when lues are all 1.0 (because D = sum_k A * B and swapping K and N in B gives D = sum_k A * B_transposed which is different).

Wait — no. If B is stored as row-major W_fp8[n][k] and the MMA expects col-major B[k][n], then reading W_fp8[n_idx][k0..k0+3] gives B[k0..k0+3][n_idx] which is 4 consecutive K values at the same N. But the MMA expects B[k_idx][n_start..n_start+3] which is 4 consecutive N values at the same K.

These are different! B[k0..k0+3][n_idx] vs B[k_idx][n_start..n_start+3]. The MMA computes D  A * B where A is [16, 32] and B is [32, 8]. If B is read incorrectly, the result is wrong.

But the working test with all B=2.0 passed because all values are the same. And the test with simple weights and ones activations passed because... let me check.

For Case 1: w_ref[k, n] = (k*N+n)%16 - 1. For n=0: all values are -1 (since 256%16=0). So W_fp8[0][k] = -1 for all k. Reading W_fp8[0][k0..k0+3] gives [-1, -1, -1, -1] which is the same as reading B[k0..k0+3][0] = [-1, -1, -1, -1]. So it works by coincidence!

For n=1: w_ref[k, 1] = (k*256+1)%16 - 1 = 1-1 = 0 for all k. So W_fp8[1][k] = 0. Reading W_fp8[1][k0..k0+3] gives [0, 0, 0, 0]. ✓

For n=2: w_ref[k, 2] = (k*256+2)%16 - 1 = 2-1 = 1 for all k. Reading W_fp8[2][k0..k0+3] gives [1, 1, 1, 1]. ✓

So for Case 1, all N-columns have uniform K-values (because (k*256)%16 = 0 for all k). So reading K-consecutive or N-consecutive gives the same result!

For Case 2 (random activations), the weights are the same as Case 1 (simple weights). So the B fragment happto give correct values because the weights are uniform along K for each N.

But for Cases 3 and 4 (random weights), the weights are NOT uniform along K. So the B fragment reads wrong values, and the result is wrong. But the test shows Cases 3 and 4 also fail...

Wait — Cases 1 and 3 pass (ones activations), Cases 2 and 4 fail (random activations). The common factor is the activations, not the weights!

So the B fragment read is correct (since Cases 1 and 3 pass with both simple and random weights). The ise is with the A fragment when activations are non-uniform.

But I verified the A fragment values match the expected values! How can the A fragment be correct but the result be wrong?

Unless the issue is with the sk=32 iteration specifically. Let me add a diagnostic for the A fragment at sk=32:

Actually, I already know the accumulator values are wrong for sk=32. Let me check if the A fragment for sk=32 reads from the correct SMEM positions.

For sk=32: col0 = 32 + (lane_id % 4) * 4 = 32, 36, 40, 44. These read from A_fp8_smem[row0 * 64 + 32] etc. These are the activations at K-positions kb + 32, kb + 36, kb + 40, kb + 44.

For kb=0: K-positions 32, 36, 40, 44. These were loaded in Phase 2 as a_fp8[mg * lda + 32] etc. ✓

Let me verify the SMEM contents for sk=32 by adding a diagnostic:

Actually, I just realized I can check this more simply. Let me verify that the accumulator for kb=0, sk=32 minus kb=0, sk=0 equals the contribution from K=32..63.

For Case 2, tid=0:

kb=0, sk=0: c[0] = -1.29
kb=0, sk=32: c[ = -3.38
Difference: -3.38 - (-1.29) = -2.09
Expected contribution from K=32..63: -sum a[0, 32:64] = -(0.45) = -0.45

The kernel gives -2.09 but expected is -0.45. The magnitude is about 4.6x too large!

Wait — sum a[0, 32:64] = 0.4453125. The kernel contribution is -2.09. But -sum a[0, 32:64] = -0.45. The kernel gives -2.09 which is NOT -0.45.

But wait — the contribution from K=32..63 should be sum_k A[0][k] * B[n_idx][k] for k=32..63. Since B[0][k] = -1 for all k (for n=0), the contribution is -sum a32:64] = -0.45.

But the kernel gives -2.09. This is wrong!

Unless the B fragment reads from the wrong SMEM positions for sk=32. Let me check:

For sk=32: k0 = 32 + (lane_id % 4) * 4 = 32, 36, 40, 44. The B fragment reads W_fp8[n_idx * 64 + 32] etc. These are the weights at K-positions kb + 32, kb + 36, kb + 40, kb + 44.

For kb=0: K-positions 32, 36, 40, 44. The weights at these positions are (k*256+0)%16 - 1 = -1 for all k (since 256%16=0). So W_fp8[0][32] = W_fp8[0][36] = W_fp8[0][40] = W_fp8[0][44] = -1. ✓

So the B fragment reads the correct values. The A fragment should read the activations at K=32,36,40,44. Let me check what those values are:

Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
So the expected contribution from K=32..63 to D[0][0] is -0.45. But the kernel gives -2.09 (the difference between sk=32 and sk=0 accumulators).

Wait — I made an error. The sk=32 value is the CUMULATIVE value (after adding the sk=32 contribution to the sk=0 result). So:

After sk=0: c = -1.29 (contrion from K=0..31)
After sk=32: c = -3.38 (after adding contribution from K=32..63)
Contribution from K=32..63: -3.38 - (-1.29) = -2.09
Expected contribution from K=32..63: -sum a[0, 32:64] = -0.45

But the kernel gives -2.09! That's about 4.6x too large!

Unless the A fragment for sk=32 reads from the wrong SMEM positions. Let me check: A_fp8_smem[row0 * 64 + col0] where col0 = 32 + (lane_id % 4) * 4.

For tid=0 (lane_id=0, row0=0): col0 = 32. Reads A_fp8_smem[0 * 64 + 32] = A_fp8_smem[32]. This was loaded in Phase 2 as a_fp8[0 * lda + kb + 32] = a_fp8[0 * 128 + 0 + 32] = a_fp8[32] = a[0, 32] = 0.8125.

But wait — the A fragment for sk=32 reads a_regs[0] from A_fp8_smem[32..35] (4 bytes). These are a[0, 32], a[0, 33], a[0, 34], a[0, 35] = [0.8125, 1.375, 0.625, 0.4375].

And a_regs[1] from A_fp8_smem[48..51] = a[0, 48..51] = [-0.34375, -0.5, -0.875, 0.40625].

The MMA computes the contribution to D[0][0] as:
sum over the 8 FP8 values in a_regs[0..1] of A_value * B_value

For n_idx=0: B values are all -1 (sin W_fp8[0][k] = -1 for all k).

So contribution = sum of 8 A values * (-1) = -(0.8125 + 1.375 + 0.625 + 0.4375 + (-0.34375) + (-0.5) + (-0.875) + 0.40625) = -(1.9375) = -1.9375

But the kernel gives -2.09 for the contribution. The expected is -1.9375. The difference is 0.15 which is within FP8 quantization error.

Wait — but the total contribution from K=32..63 should be -sum a[0, 32:64] = -0.45. But the 8 values I listed only cover K=32..35 and K=48..51. The remaining K values (36..47 and 52..63) are coved by other threads in the warp!

The MMA computes D[0][0] as the sum over ALL 32 K values (k=32..63), not just the 8 values held by thread 0. The warp collectively holds all 32 K values, and the MMA instruction sums them internally.

So the contribution from K=32..63 should be -sum a[0, 32:64] = -0.45. But the kernel gives -2.09. This means the MMA is computing the wrong sum!

But my standalone test showed the MMA works correctly with varied A values. So the issue must be in how our kernel loads the data into SMEM or how the warp collaboratively reads the data.

Actually, I think I finally found it! The issue is with the B fragment stride. In our kernel, W_fp8 is stored as [N][K] with stride kTileK = 64. The B fragment reads W_fp8[n_idx * 64 + k0]. For sk=32, k0 = 32, 36, 40, 44 and k1 = 48, 52, 56, 60. These read from W_fp8[n_idx * 64 + 32] etc.

But the MMA expects the B fragment to be in col-major format with stride 8 (N dimension). Our stride is 64 (K dimension). The B fragment reads 4 consecutive bytes which are 4 consecutive K values at the same N. In col-major, these should be 4 consecutive N values at the same K.

But wait — I proved earlier that the B fragment read is correct because:

The working test with col-major B (stride 8) passes
Our kernel's B fragment (stride 64) reads 4 consecutive K values
The MMA expects 4 consecutive N values (not K values!)
So our B fragment reads the wrong values! But Cases 1 and 3 pass because the weights are uniform along K (all values in a given N-column are the sam.

For Case 1 (simple weights): W_fp8[n][k] = (k*256+n)%16 - 1. For a fixed n, the value depends on k. For n=0: (k*256)%16 - 1 = -1 (constant). For n=1: (k*256+1)%16 - 1 = 0 (constant). For n=2: (k*256+2)%16 - 1 = 1 (constant).

So for each n, all K values are the same! Reading K-consecutive values gives the same result as reading N-consecutive values.

For Cases 3 and 4 (random weights): the values vary along both K and N. Reading K-consecutive gives different values than N-consecutive. But Cases 3 and 4 still pass with ones activations!

Wait — Cases 3 and 4 use random weights. Let me check: for Case 3 (random weights, ones activations), the test says PASS. But the B fragment reads wrong values. How can it pass?

Let me think... With a = ones(M, K), D[row][col] = sum_k W[k][col]. The B fragment for sk=0 reads W_fp8[n_idx][k0..k0+3] (4 consecutive K values at the same N). The MMA computes D += A * B where A values are all 1.0. So D = sum of all B values = sum of W_fp8[n_idx][k0..k0+3] for all threads.

Buthe correct computation should be: D[row][col] = sum_k W[k][col]. The B fragment should read B[k_idx][n_start..n_start+3] (4 consecutive N values at the same K). But instead it reads W_fp8[n_idx][k0..k0+3] (4 consecutive K values at the same N).

These are different! But with ones activations, the result happens to be the same because the MMA sums over ALL K values for each output position, and the total sum is the same regardless of the order of summation.

Wait — no. The MMA computes D += A * B where A i[16, 32] and B is [32, 8]. The result D is [16, 8]. Each element D[i][j] = sum_k A[i][k] * B[k][j].

If the B fragment reads W_fp8[n_idx][k0..k0+3] instead of B[k_idx][n_start..n_start+3], then the MMA computes:
D[i][j] = sum_k A[i][k] * W_fp8[j][k] (since n_idx = j and k0 = k)

But the correct computation should be:
D[i][j] = sum_k A[i][k] * B[k][j] = sum_k A[i][k] * W_fp8[j][k]

Wait — B[k][j] = W_fp8[j][k] (since we store weights as W_fp8[n][k]). So B[k][j] = W_fp8[j][k]. And the B fragment reads W_fp8_idx][k0..k0+3] where n_idx = wn + lane_id/4 = j and k0 = sk + (lane_id%4)*4 = k.

So the B fragment reads W_fp8[j][k..k+3] which is B[k..k+3][j]. This is 4 consecutive K values at the same N index j. In col-major B, this should be 4 consecutive N values at the same K index.

But B[k..k+3][j] = W_fp8[j][k..k+3]. The correct col-major read should be B[k][j..j+3] = W_fp8[j..j+3][k].

These are different! W_fp8[j][k..k+3] vs W_fp8[j..j+3][k]. The first reads along K, the second reads along N.

With ones activations (A=1.0), D[i][j] = sum_k B[k][j]. If B is read incorrectly as B[k..k+3][j] instead of B[k][j..j+3], the sum over k still gives the same total because we're summing over all k values. The issue is that the wrong B values are paired with the wrong A values.

With A=1.0, D[i][j] = sum_k B[k][j] = sum_k W_fp8[j][k]. The MMA computes D[i][j] = sum_k A[i][k] * B_wrong[k][j] = sum_k 1.0 * W_fp8[j][k] = sum_k W_fp8[j][k]. This is the same! ✓

With random A, D[i][j] = sum_k A[i][k] * B[k][j]. The MMA computeD[i][j] = sum_k A[i][k] * W_fp8[j][k]. But the correct computation is D[i][j] = sum_k A[i][k] * W_fp8[j][k] (since B[k][j] = W_fp8[j][k]).

Wait — these are the SAME! B[k][j] = W_fp8[j][k]. So reading W_fp8[j][k..k+3] gives B[k..k+3][j], which is 4 consecutive K values of B at the same N. But the MMA expects 4 consecutive N values of B at the same K.

The MMA computes: D[i][j] += sum over the 4 values in b_regs[0] of A[i][k] * b_value. If b_regs[0] contains B[k..k+3][j] (4 K values at same N), then:
D[i][ += A[i][k]*B[k][j] + A[i][k+1]*B[k+1][j] + A[i][k+2]*B[k+2][j] + A[i][k+3]*B[k+3][j]

But the MMA expects b_regs[0] to contain B[k][j..j+3] (4 N values at same K):
D[i][j] += A[i][k]*B[k][j] + A[i][k]*B[k][j+1] + ...

These are completely different computations! The first sums over K for a fixed (i,j), the second sums over N for a fixed (i,k).

So the B fragment is DEFINITELY wrong. But Cases 1 and 3 pass because:

Case 1: weights are uniform along K → B[k][j] = B[k+1][j] = ... → both read patterns give same values
Case 3: weights are random but A=1.0 → the sum over k of B[k][j] is the same regardless of the read pattern
Cases 2 and 4 fail because A is random and weights are non-uniform along K.

The fix: change the B fragment to read col-major values. Since our SMEM stores W_fp8[n][k] (row-major), we need to read W_fp8[n..n+3][k] (4 consecutive N values at the same K) instead of W_fp8[n][k..k+3] (4 consecutive K values at the same N).

But these are NOT consecutive in SMEM (stride = kTileK = 64). We nd to gather individual bytes.

Let me implement this fix:

Edited
w4a8_fp8_qmma.cu+17-7
Now remove the diagnostic printf:

Edited
w4a8_fp8_qmma.cu+0-4
Wait, I also need to check bounds: n_start + 16 + 3 could exceed kTileN - 1 = 127. For wn = 120 and (lane_id % 4) * 4 = 12: n_start = 132. n_start + 16 + 3 = 151 > 127. This would be out of bounds!

But the B fragment only covers 8 N-rows. wn = ns * 8 for ns = 0..15. For wn = 120 (ns=15): n_start = 120 + (lane_id % 4) * 4. For lane_id % 4 = 3: n_start = 132. n_start + 3 = 135 > 127.

But n_start should be within the 8 N-rows of the B fragment. The B fragment has 8 N-rows (indices 0..7). n_start is the starting N index within the fragment. For lane_id % 4 = 3: n_start = 12, but the fragment only has 8 N-rows!

This means (lane_id % 4) * 4 should only go up to 4 (covering N positions 0..3 and 4..7). Values 8 and 12 are outside the 8 N-rows!

But the PTX ISA says b[0] = B[t/4][(t%4)*4..+4]. For t%4 = 0, 1, 2, 3: (t%4)*4 = 0, 4, 8, 12. But B only has 8 N values (0..7). So offsets 8 and 12 are invalid!

This means the B fragment layout must be different. Let me reconsider.

For col-major B with shape [32, 8]:

First index: K (0..31)
Second index: N (0..7)
Thread t provides:

b[0] = 4 consecutive values = 4 bytes
b[1] = 4 consecutive values = 4 bytes
Total: 8 bytes = 8 FP8 values per thread
32 threads × 8 values = 256 FP8 values = 32 × 8 = B's total size ✓
The PTX ISA says b[0] = B[t/4][(t%4)*4..+4]. The first index t/4 ranges 0..7. The second index (t%4)*4 ranges , 8, 12.

For col-major B, the first index is K and the second is N. So:

B[t/4][(t%4)*4..+4] = B[K=t/4][N=(t%4)*4..+4]
But N only has 8 values! With offsets 0, 4, 8, 12, offsets 8 and 12 are out of range!

Unless B is NOT [32, 8] but rather [32, 16]! Or unless the first index is N and the second is K (row-major interpretation for the fragment).

Let me reconsider. The MMA instruction is row.col:

A: row-major
B: col-major
For row-major A [16, 32]: A[i][j] at i*32 + j. Thread t: a[0] = A[t/4][(t%4)*4..+4]. First index t/4 = row 0..7, second index (t%4)*4 = col 0,4,8,12. A has 32 columns, so offsets 0,4,8,12,16,20,24,28 are valid (covered by a_regs[0] and a_regs[1]).

For col-major B [8, 32]: B[i][j] at i + j*8 (col-major storage). Thread t: b[0] = B[t/4][(t%4)*4..+4]. First index t/4 = row 0..7, second index (t%4)*4 = col 0,4,8,12.

Wait — for col-major B with shape [8, 32], the first dimension is 8 and the second is 32. So:

First index t/4 ranges 0..7 → valid for first dimension (8)
Second index (t%4)*4ges 0, 4, 8, 12 → but second dimension is 32, so offsets 0..12 are valid!
So B has shape [8, 32] where the first index is the "row" (N dimension in col-major) and the second is the "column" (K dimension in col-major).

In col-major storage, B[row][col] is at position row + col * 8. So:

B[0][0] at position 0
B[0][1] at position 8
B[1][0] at position 1
...
Thread t: b[0] = B[t/4][(t%4)*4..+4] = 4 consecutive values at row t/4, columns (t%4)*4 to (t%4)*4+3.

In col-major storage, these are at positions:

(t) + ((t%4)*4 + 0) * 8 = t/4 + (t%4)*32
(t/4) + ((t%4)*4 + 1) * 8 = t/4 + (t%4)*32 + 8
(t/4) + ((t%4)*4 + 2) * 8 = t/4 + (t%4)*32 + 16
(t/4) + ((t%4)*4 + 3) * 8 = t/4 + (t%4)*32 + 24
These are NOT consecutive in memory! They are spaced 8 bytes apart (stride 8 in the column dimension).

So memcpy(&b_regs[0], &smem_b[t/4 + (t%4)*32], 4) would read 4 consecutive bytes at positions t/4 + (t%4)*32 to t/4 + (t%4)*32 + 3. But the 4 FP8 values should be at positions t/4 + (t%4)*32, t/4 + (t%4)*32 + 8, t/4 + (t%4)*32 + 16, t/4 + (t%4)*32 + 24.

These are NOT consecutive! So the 4 bytes in b_regs[0] are NOT 4 consecutive FP8 values. They must be gathered from 4 different positions.

But earlier, my standalone test with col-major B stored as smem_b[k * 8 + n] and read as memcpy(&b_regs[0], &smem_b[k * 8 + n_start], 4) passed! How?

Let me re-examine. In the standalone test:

B stored as smem_b[k * 8 + n] (col-major: B[k][n] at k*8+n)
B fragment reads smem_b[k_idx * 8 + n_start] where k_idx = lane_id / 4 and n_start = (lane_id % 4) * 4
For lane_id=0: k_idx=0, n_start=0. Reads smem_b[0..3] = B[0][0], B[0][1], B[0][2], B[0][3] = 4 consecutive N values at K=0. ✓

For lane_id=1: k_idx=0, n_start=4. Reads smem_b[4..7] = B[0][4], B[0][5], B[0][6], B[0][7] = 4 consecutive N values at K=0. ✓

For lane_id=4: k_idx=1, n_start=0. Reads smem_b[8..11] = B[1][0], B[1][1], B[1][2], B[1][3] = 4 consecutive N values at K=1. ✓

So the standalone test reads 4 consecutive N values at the same K. This is correct for col-major B!

Now in onel, we need to read the same thing from row-major W_fp8. W_fp8[n][k] = B[k][n]. So B[k][n..n+3] = W_fp8[n..n+3][k]. These are at positions (n+0)*kTileK + k, (n+1)*kTileK + k, (n+2)*kTileK + k, (n+3)*kTileK + k. NOT consecutive!

So the gather code I wrote is correct! Let me verify the bounds:

k_idx = sk + lane_id / 4 ranges sk to sk + 7. For sk=0: 0..7. For sk=32: 32..39. But kTileK = 64, so k_idx max = 39 < 64. ✓
n_start = wn + (lane_id % 4) * 4 ranges wn to wn + 12. For wn = ns * 8 with ns = 0..15: wnax = 120. n_start max = 132. n_start + 3 max = 135. But kTileN = 128!
So n_start + 3 can be 135 which is > 127! This is out of bounds!

The fix: n_start should be capped at kTileN - 4. But actually, the B fragment only covers 8 N-rows. n_start should be within wn to wn + 7. Since (lane_id % 4) * 4 gives 0, 4, 8, 12, and the fragment has 8 N-rows, only offsets 0 and 4 are valid!

Wait — b[0] covers N positions (t%4)*4..+4 = 4 consecutive N values. With (t%4)*4 = 0, 4: covers N=0..3 and N=4..7. With (t%4)*4 8, 12: covers N=8..11 and N=12..15 which are outside the 8 N-rows!

But the PTX ISA says b[0] = B[t/4][(t%4)*4..+4]. If B has 8 N-rows, then (t%4)*4 can only be 0 or 4. But t%4 gives 0, 1, 2, 3, so (t%4)*4 = 0, 4, 8, 12. This means offsets 8 and 12 are invalid for an 8-wide N dimension!

Unless B has 16 N-rows! Maybe m16n8k32 actually produces a 16×16 output (M=16, N=16) and B has shape [32, 16]?

Let me check: m16n8k32 means M=16, N=8, K=32. But maybe the N in the instruction name doesn't match the actua B fragment shape?

Actually, looking at the PTX ISA more carefully: for m16n8k16 with f16, the output is [16, 8]. For m16n8k32 with fp8, the output should also be [16, 8]. But the B fragment has 2 uint32_t registers per thread, which is 8 FP8 values per thread × 32 threads = 256 FP8 values. If B has shape [K, N] = [32, 8], that's 256 values. ✓

But with (lane_id % 4) * 4 giving 0, 4, 8, 12, and only 8 N values, offsets 8 and 12 are invalid!

Unless the B fragment has shape [K/4, N] = [8, 8] for each thr group. Let me reconsider:

Thread t: b[0] = B[t/4][(t%4)*4..+4]. With 32 threads:

t/4 ranges 0..7: 8 groups
(t%4)*4 ranges 0, 4, 8, 12: 4 positions per group
Each group covers K positions t/4 to t/4+3 (4 K values). And 4 N positions (t%4)*4 to (t%4)*4+3. Total: 8 × 4 × 4 = 128 values per b_regs[0] and b_regs[1], total 256. ✓

But N only has 8 values. With 4 positions 0, 4, 8, 12, positions 8 and 12 are outside!

Unless the B fragment is actually shaped [32, 16] (K=32, N=16), and the output is [16, 16]16, N=16). But the instruction is called m16n8k32, not m16n16k32.

I think the issue is that b[0] and b[1] together cover 8 consecutive N positions (0..7 and 8..15), but the instruction is m16n8k32 with N=8, so only the first 8 N positions are valid. The (t%4)*4 values 8 and 12 are for N positions 8..11 and 12..15 which are outside the 8-wide N tile!

But the PTX ISA specifies this layout for all threads, even those with (t%4)*4 >= 8. Those threads would read outside the B fragment!

This can't be right. Let me check the CUDA PTX ISA documentation more carefully.

Actually, I think I've been misunderstanding the layout. Let me reconsider. For m16n8k32 with col-major B:

The B fragment has shape [32, 8] in the GEMM sense: 32 K-values and 8 N-values. But "col-major" means the first index is the column (K) and the second is the row (N). So B[K][N] at position K*8 + N.

Thread t: b[0] = B[t/4][(t%4)*4..+4]. The first index t/4 ranges 0..7, which is the K index (0, 4, 8, ..., 28). The second index (t%4)*4 ranges 0, 4, 8, 12, which is the N start position.

But N only has 8 values! So N start positions 8 and 12 are invalid!

Unless... the B fragment is NOT [32, 8] but [8, 32] (row-major interpretation). Let me re-read the PTX ISA:

For mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32:

.row.col: A is row-major, B is column-major
m16n8k32: M=16, N=8, K=32
For column-major B with shape [K, N] = [32, 8]:

B is stored with K as the inner dimension (column-major means columns are contiguous)
B[k][n] at position k + n * 32 (K-inner, N-outer)
Wait — "column-major" means the first index varies the fastest. For B[k][n], the first index is k (row in the GEMM sense, which is the K dimension), and the second is n (column in the GEMM sense, which is the N dimension).

In column-major storage, B[k][n] is at position k + n * ld where ld is the leading dimension (number of rows = K = 32). So B[k][n] at k + n * 32.

Thread t: b[0] = B[t/4][(t%4)*4..+4]. First index t/4 = K (0..7). Second index (t%4)*4..+4 = N start position.

In cumn-major storage at k + n * 32:

B[t/4][(t%4)*4 + 0] at position t/4 + ((t%4)*4 + 0) * 32 = t/4 + (t%4)*128
B[t/4][(t%4)*4 + 1] at position t/4 + ((t%4)*4 + 1) * 32 = t/4 + (t%4)*128 + 32
B[t/4][(t%4)*4 + 2] at position t/4 + ((t%4)*4 + 2) * 32 = t/4 + (t%4)*128 + 64
B[t/4][(t%4)*4 + 3] at position t/4 + ((t%4)*4 + 3) * 32 = t/4 + (t%4)*128 + 96
These are NOT consecutive! Stride 32.

But the PTX ISA says b[0] is a single uint32_t (4 bytes). How can 4 non-consecutive bytes be packed into a uint32_t?

Unless the storage is different. Let me reconsider. Maybe for the MMA instruction, the B fragment is NOT in standard column-major storage, but in a special format where the 4 values in b_regs[0] ARE consecutive.

Looking at the PTX ISA documentation for mma.sync:

The layout of the fragments is optimized for the instruction. The programmer must store data into the fragments using the provided load instructions (e.g., ldmatrix).

So the fragment layout is NOT standard row/column-major! It's a special format optimized for the MMA instruction. The programmer must use ldmatrix to load data into the fragments.

But we're using inline PTX and loading data manually with memcpy. The fragment layout is defined by the PTX ISA and must be followed exactly.

Let me look at how ldmatrix works for the B fragment. The ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 instruction loads a 8×8 matrix from SMEM into 4 uint32_t registers, with optional transpose.

For the B fragment of m16n8k32, the layout might be different from what Iassumed. The ldmatrix instruction handles the format conversion.

Since we're not using ldmatrix, we need to know the exact register layout. Let me check the PTX ISA specification for the exact bit layout of the B fragment.

Actually, I think the key insight is that the 4 bytes in b_regs[0] ARE 4 consecutive bytes in memory, but they represent 4 FP8 values from different (k, n) positions according to the MMA's internal layout.

From my standalone test, the correct read for col-major B stored as B[k][n] at k*8 + n is:


memcpy(&b_regs[0], &smem_b[k_idx * 8 + n_start], 4);
where k_idx = lane_id / 4 and n_start = (lane_id % 4) * 4.

This reads 4 consecutive bytes at positions k_idx*8 + n_start to k_idx*8 + n_start + 3. In col-major storage, these are B[k_idx][n_start..n_start+3] = 4 consecutive N values at the same K.

For our kernel, we need to produce the same register values. Our SMEM stores W_fp8[n][k] (row-major). B[k][n] = W_fp8[n][k]. So B[k_idx][n_start..n_start+3] = W_fp8[n_start..n_start+3][k_idxese are at positions (n_start+0)*kTileK + k_idx, (n_start+1)*kTileK + k_idx, etc. — NOT consecutive.

So my gather code is correct! The bounds issue is: n_start + 16 + 3 can exceed kTileN. Let me fix the bounds:

For wn = ns * 8 with ns = 0..15: wn max = 120. n_start = wn + (lane_id % 4) * 4. For lane_id % 4 = 3: n_start = 120 + 12 = 132. n_start + 16 + 3 = 151 > 127.

But wait — b[1] covers N positions n_start + 16 to n_start + 19. For n_start = 132: N positions 148..151. But the B fragment only has 8 ws!

The issue is that (lane_id % 4) * 4 should only be 0 or 4 for an 8-wide N tile. Values 8 and 12 are outside. But the PTX ISA specifies this layout for all threads.

Looking at my standalone test: b_regs[1] reads from smem_b[k_idx * 8 + n_start + 16]. For n_start = 12: reads from position k_idx * 8 + 28. But the SMEM has 32 elements per K-row (8 N-values × 4 bytes). Position k_idx * 8 + 28 = k_idx * 8 + 28. For k_idx = 0: position 28. But the SMEM has 32 elements per K-row. So position 28 is valid (witin 0..31).

Wait — the standalone test's SMEM for B has 32 * 8 = 256 elements (col-major [32][8]). Each K-row has 8 elements. So k_idx * 8 + n_start + 16 for n_start = 12: k_idx * 8 + 28. For k_idx = 0: position 28, which is within 0..31 for the first K-row. But b[1] should read N positions n_start + 16 to n_start + 19 = 28..31. But N only has 8 values (0..7)!

So b[1] reads outside the B fragment! But the standalone test passed because all B values were 2.0 (uniform).

This means my understanding of the fragment layout is still wrong. Let me look at this differently.

For m16n8k32 with FP8:

A fragment: 4 uint32_t per thread, 32 threads = 512 FP8 values = 16 × 32 = A's size ✓
B fragment: 2 uint32_t per thread, 32 threads = 256 FP8 values = ? × ?
If B has shape [32, 8]: 32 × 8 = 256 ✓

But the 4 bytes in b_regs[0] represent 4 FP8 values. With (t%4)*4 = 0, 4, 8, 12, the N positions covered are 0..3, 4..7, 8..11, 12..15. But N only has 8 values!

Unless the B fragment covers N=0..15 (16 values), and thuction is actually m16n16k32 for the B fragment but m16n8k32 for the output!

No, that doesn't make sense. Let me check the CUDA programming guide.

Actually, I think the answer is simpler. The m16n8k32 instruction with FP8 e4m3 produces a [16, 8] output. But the B fragment has shape [32, 16] internally, and only 8 of the 16 N values contribute to the output. The remaining 8 N values are discarded (or used for something else).

This would explain why (t%4)*4 gives offsets 0, 4, 8, 12 — they cover N=0..15.nd only N=0..7 contribute to the output.

If this is the case, then our kernel's B fragment should read all 16 N values, even though only 8 contribute. But our SMEM only has kTileN = 128 N values, and for wn = 120, n_start + 16 + 3 = 151 > 127.

The fix: pad the SMEM to handle the extra N values, or restrict wn to ensure n_start + 19 < kTileN.

Actually, the simplest fix is to ensure wn + 19 < kTileN. Since wn = ns * 8, we need ns * 8 + 19 < 128, so ns < 13.6. So ns can only go up to 13, not 15. But the kernel iterates ns = 0..15.

Wait — but the B fragment only needs 8 N-rows for the output. The extra N values (8..15) might be needed for the internal computation but don't contribute to the output. If we read out of bounds, we get garbage values that might affect the output through the MMA's internal computation.

The fix: allocate extra SMEM for W_fp8 to handle the 16 N-rows needed by the B fragment. Or change the SMEM layout.

Actually, the simplest fix is to change the SMEM stride for W_fp8 from kTileN tkTileN + 8 (or more), ensuring that n_start + 19 is always within bounds. But this wastes SMEM.

A better fix: store W_fp8 with stride kTileN_padded = ((kTileN + 15) / 16) * 16 = 128 (already 128, which is a multiple of 16). But n_start + 19 max = 120 + 12 + 19 = 151. We need at least 152 elements per N-row.

So we need kTileN >= 152 or we need to pad to at least 152. But kTileN = 128. We can't change it without changing the tile size.

Alternative: restrict the N-tile loop to ns < 14 (so wn < 112). Then n_start + 19 <= 112 + 12 + 19 = 143 < 152. But we still need 152 elements per N-row.

Actually, the simplest approach: change the W_fp8 SMEM stride from kTileN to kTileN + 24 (padding of 24). This ensures n_start + 19 is always within bounds.

But changing the stride affects all SMEM accesses. Let me think of a better approach.

Actually, the cleanest fix is to store W_fp8 in col-major format (as B[k][n] at k * kTileN + n), which allows the B fragment to read 4 consecutive bytes. This requires changing the dequant storage format and the B fragment read.

Let me implement this:

Read w4a8_fp8_qmma.cu, lines 40 to 80

I see the SMEM layout: W_fp8 has kTileN * kTileK = 128 * 64 = 8192 elements. A_fp8_smem = W_fp8 + 8192. Total SMEM = 16384 bytes = 16KB.

The simplest fix: store W_fp8 in col-major format W_fp8[k * kTileN + n] instead of row-major W_fp8[n * kTileK + k]. This way, the B fragment can read 4 consecutive bytes for 4 consecutive N values at the same K.

But the SMEM size stays the same (kTileK * kTileN). And the B fragment reads W_fp8[k_idx * kTileN + n_start] which is 4 consecutive bytes at K-index k_idx, N-indices n_start..n_start+3. ✓

The bounds issue: k_idx * kTileN + n_start + 3 max = (sk + 7) * 128 + (wn + 12) + 3 = 39 * 128 + 135 = 5112. W_fp8 has kTileK * kTileN = 64 * 128 = 8192 elements. ✓

And (k_idx + 16) * kTileN + n_start + 3 max = (39 + 16) * 128 + 135 = 55 * 128 + 135 = 7175. Still within 8192. ✓

This approach works! Let me implement it:

Edited
w4a8_fp8_qmma.cu+4-2
Now update the Bent to read from col-major storage:

Edited
w4a8_fp8_qmma.cu+6-15
Now commit, push, build, and test:

Executed commands in the terminal
Still failing! The col-major storage change didn't fix it. Let me think about why...

Actually, the issue might be that the B fragment with col-major storage still reads incorrectly. Let me trace through the values.

For col-major W_fp8[k][n] at k * kTileN + n:

b_regs[0] reads from W_fp8[k0 * kTileN + n_start] for 4 consecutive bytes
These are W_fp8[k0][n_start..n_start+3] = B[k0][n_start..n_start+3]
For lane_id=0, sk=0, wn=0: k0=0, n_start=0. Reads W_fp8[0][0..3] = weights at (k=0, n=0..3). ✓

For lane_id=1, sk=0, wn=0: k0=0, n_start=4. Reads W_fp8[0][4..7] = weights at (k=0, n=4..7). ✓

For lane_id=4, sk=0, wn=0: k0=1, n_start=0. Reads W_fp8[1][0..3] = weights at (k=1, n=0..3). ✓

This looks correct. But the test still fails. Let me check the k0 + 16 bound. For sk=32, lane_id=28: k0 = 32 + 7 = 39. k0 + 16 = 55. W_fp8[55 * 128 + n_start] — 55 * 128 = 7040. W_fp8 hasements. ✓

Wait — the issue might be with the k0 range. k0 = sk + lane_id / 4. For sk=32: k0 ranges 32..39. k0 + 16 ranges 48..55. But kTileK = 64, so K indices 48..55 are valid. ✓

Hmm, everything looks correct. Let me add a quick diagnostic to check the B fragment values:

Updated w4a8_fp8_qmma.cu and ran commands
The variables are scoped inside the block. Let me fix:

Updated w4a8_fp8_qmma.cu and ran commands
Wait — n_start has values 0, 4, 8, 12! But the B fragment only has 8 N-rows! For n_start n_start = 12, the reads are at N positions 8..11 and 12..15 which are OUTSIDE the valid N range!

But the values look reasonable: b_regs[0]=0x5251504e(7) for n_start=8. The first byte is 0x4e = 7 (FP8). This is B[0][8] which is outside the 8 N-rows!

The fact that the values are non-zero and look like valid weights suggests that the SMEM has data at those positions (from the next N-tile). But these values should NOT be in the B fragment for the first N-tile!

This confirms that the B fragment layout covers N positions 0..15, not 0..7. The MMA instruction uses N=16 internally, and the m16n8k32 name refers to the output size (N=8), not the internal B fragment size (N=16).

So the B fragment has 16 N-rows, and n_start = 0, 4, 8, 12 are all valid. The 4 bytes in b_regs[0] cover N positions n_start..n_start+3, which are within 0..15. ✓

But our SMEM only has kTileN = 128 N-rows per tile. For wn = ns * 8 with ns = 0..15:

n_start = wn + (lane_id % 4) * 4
wn max = 120
n_start max = 120 + 12 = 132
n_start + 3 max 135
But kTileN = 128. So n_start + 3 can be 135 > 127! This reads out of bounds!

The fix: ensure n_start + 19 < kTileN (for b_regs[1] which reads n_start + 16 to n_start + 19). Since n_start max = wn + 12, we need wn + 12 + 19 < kTileN, so wn < kTileN - 31 = 97. With wn = ns * 8, ns < 12.1. So ns can only go up to 11, not 15!

But the kernel iterates ns = 0..15. For ns >= 12, wn >= 96, and n_start + 19 can exceed 127!

This means the kernel has a bounds bug for the last few N-tiles. But the test uses N = 256, and kTileN = 128, so there are 2 N-tiles (nb=0, 1). For nb=0: n0 = 0, wn max = 120, n_start + 19 max = 139 > 127. Out of bounds!

But for nb=1: n0 = 128, ng = 128 + n. For wn = 120: ng = 248. n_start + 19 = 139. ng = n0 + n_start + 19 = 128 + 139 = 267 > N = 256. So the bounds check ng < N in Phase 1 would prevent writing to these positions. But the SMEM still has kTileN = 128 elements, so reading W_fp8[139] is out of bounds!

Wait — W_fp8[k * kTileN + n] where kTileN = 128. For k0 = 0, n_start = 132:_fp8[0 * 128 + 132] = W_fp8[132]. But W_fp8 has kTileK * kTileN = 64 * 128 = 8192 elements. Index 132 is within bounds!

But W_fp8[132] was written by Phase 1 as W_fp8[k * kTileN + n] where k = 0 and n = 132. But n ranges 0..127 in Phase 1 (since i < kTileN * kTileK and n = i / kTileK). So W_fp8[132] was NOT written by Phase 1 — it contains garbage!

This is the bug! The B fragment reads from SMEM positions that were never written!

The fix: increase the W_fp8 SMEM allocation to handle the extra N positio needed by the B fragment. The B fragment needs N positions up to n_start + 19 where n_start = wn + 12 and wn = ns * 8. For ns = 15: wn = 120, n_start = 132, n_start + 19 = 151. So we need at least 152 N elements per K-row.

Change kTileN to kTileN_padded = 152 for the W_fp8 allocation only. Or better, just increase the SMEM allocation.

Actually, the simplest fix: change the SMEM stride for W_fp8 from kTileN to kTileN + 24 (padding of 24). But this changes all accesses.

Even simpler: just pad the W_fp8 allocation to kTileN + 24 elements per K-row. The SMEM size becomes kTileK * (kTileN + 24) + kTileM * kTileK = 64 * 152 + 128 * 64 = 9728 + 8192 = 17920 bytes = 17.5KB. Still within the 48KB limit.

But this requires changing the stride everywhere. Let me think of a cleaner approach.

Actually, the cleanest fix: limit the N-tile loop to ensure n_start + 19 < kTileN. Since n_start = wn + (lane_id % 4) * 4 and wn = ns * 8, we need ns * 8 + 12 + 19 < kTileN, so ns < (128 - 31) / 8 = 12.1. So ns max = 11, covering 12 N-tiles × 8 = 96 N-rows per block.

But this means each block only covers 96 N-rows instead of 128. We'd need more blocks.

Alternatively, just allocate extra SMEM space. Let me modify the SMEM allocation:

Edited
w4a8_fp8_qmma.cu+3-1
Now update the SMEM allocation size:

Edited
w4a8_fp8_qmma.cu+1-1
Now remove the diagnostic and update the kernel to use the padded stride:

Edited
w4a8_fp8_qmma.cu+0-5
Now commit and test:

Executed commands in the terminal
kWPadN is defined inside the kernel function bt kSmemBytes is outside. Let me move it:

Updated w4a8_fp8_qmma.cu and ran commands
Still failing! The SMEM padding didn't help. Let me think more carefully.

Actually, I just realized the SMEM padding doesn't help because the dequant writes to W_fp8[k * kTileN + n] where kTileN = 128. The padded area (n >= 128) is never written, so reading from it gives garbage. The padding only prevents out-of-bounds memory access, but the values are still garbage.

The real issue is that the B fragment reads N positions that exceed the tile's N range. For wn = 120 and n_start = 132: the B fragment reads W_fp8[k0][132..135] which is in the padded area (uninitialized).

The fix should be to ensure the B fragment reads only within the valid N range. Since the B fragment covers N positions 0..15 within each N-tile, and our tile has kTileN = 128 N-rows divided into 16 N-tiles of 8 rows each, the B fragment for N-tile ns covers N-rows ns*8 to ns*8 + 15. But the N-tile only has 8 rows!

So the B fragment reads 16 N-rows but the tile only has 8 N-rows per N-tile. The extra 8 N-rows (positions 8..15 within the fragment) overlap with the next N-tile!

This means the B fragment layout overlaps between adjacent N-tiles. The MMA instruction uses 16 N-rows internally, and adjacent N-tiles share 8 of those rows.

For m16n8k32, the output is [16, 8]. The B fragment is [32, 16] internally. The MMA computes:
D[16][8] = A[16][32] @ B[32][16]

But only the first 8 N-columns of D are used (the n8 in the instruction name). The last 8 are discarded.

So the B fragment reads 16 N-rows, and the N-tiles should be spaced 8 apart (not 16). The overlap is intentional!

For ns=0: B fragment covers N=0..15, output covers N=0..7
For ns=1: B fragment covers N=8..23, output covers N=8..15
For ns=2: B fragment covers N=16..31, output covers N=16..23
...

So wn = ns * 8 is correct. The B fragment reads N positions wn + n_start to wn + n_start + 19 where n_start = (lane_id % 4) * 4 ranges 0, 4, 8, 12. For wn = 120 (ns=15): N positions 120..139. But the tile only has N=0..127. Positions 128..139 are out of bounds!

The fix: for the last N-tile, the B fragment reads beyond the tile boundary. We need to either:

Zero-pad the SMEM for out-of-range N positions
Handle the bounds in the B fragment read
The simplest fix: zero-initialize the W_fp8 SMEM padding area. But this adds overhead.

Better fix: adjust the N-tile iteration to only go up to ns < 14 (so wn < 112), and handle the remaining N-rows separately. But this complicates the kernel.

Simplest correct fix: pad W_fp8 to kTileN + 16 and zero-initialize the padding in Phase 1. Let me update Phase 1 to write zeros for out-of-range N positions:

Edited
w4a8_fp8_qmma.cu+6-6
Now update the SMEM pointer and all W_fp8 accesses to use the padded stride:

Updated w4a8_fp8_qmma.cu
Now update the B fragment to use the padded stride:

Updated w4a8_fp8_qmma.cu and ran commands
kWStrideN is in the kernel scope. Let me move it:

Updated w4a8_fp8_qmma.cu and ran commands+1-0
Still failing! The padded stride approach didn't fix it. Let me reconsider the entire approach.

Looking at this more carefully, I think the fundamental issue is that the B fragment reads 4 consecutive bytes which are 4 consecutive N values at the same K. In our col-major storage W_fp8[k * kWStrideN + n], these ARE consecutive. So the read should be correct.

But wait — the B fragment has shape [32, 16] (K=32, N=16), not [32, 8]. The output is [16, 8]. The MMA computes D[16][8] = A[16][32] @ B[32][16], but only the first 8 N-columns of D are used.

For the B fraent to read N positions 0..15 (16 values), the threads need to cover 16 N positions. With 32 threads and 2 uint32_t per thread:

Each thread contributes 8 FP8 values (2 × 4 bytes)
32 × 8 = 256 FP8 values
256 / 16 = 16 K values per thread group
Wait — 256 / 16 = 16, not 32. So B has shape [16, 16] (K=16, N=16)?

But the instruction is m16n8k32 with K=32. So B should have K=32.

Let me recalculate: 256 FP8 values / 32 K-values = 8 N-values. So B is [32, 8]. But the fragment reads 16 N-values per K-row (n_t + 16 + 3 = n_start + 19, covering 20 N positions).

This doesn't add up. Unless the fragment layout is more complex than I think.

Let me just check what the correct B fragment read is by examining the working standalone test more carefully. In that test:

B is stored as B[k][n] at k * 8 + n (col-major, 8 N-values per K-row)
B fragment reads smem_b[k_idx * 8 + n_start] where k_idx = lane_id / 4 and n_start = (lane_id % 4) * 4
For lane_id=0: reads smem_b[0..3] = B[0][0..3] ✓
For lane_id=1: reads smem_b[47] = B[0][4..7] ✓
For lane_id=2: reads smem_b[8..11] = B[0][8..11] — but N only has 8 values!

But the test passed! So B[0][8..11] must be valid. This means B has at least 12 N-values per K-row.

But the SMEM only has 8 N-values per K-row (256 elements / 32 K-rows = 8). So smem_b[8..11] reads from the NEXT K-row's data!

This means the B fragment reads across K-row boundaries! The stride is 8, and the fragment reads 4 bytes from k*8 + n_start to k*8 + n_start + 3. For n_start = 8: reads from k*8 + 8 to + 11. But k*8 + 8 = (k+1)*8 + 0 which is the start of the next K-row!

So the B fragment reads 4 bytes that span two K-rows! The first 2 bytes are from K-row k at N positions 8..9, and the next 2 bytes are from K-row k+1 at N positions 0..1.

But in our col-major storage with stride kWStrideN = 152, k*152 + n_start for n_start = 8 reads from k*152 + 8. The next byte is at k*152 + 9, not (k+1)*152 + 0. So our stride is correct for reading within a single K-row.

But the B fragment expects the data to be packed with stride 8 (8 N-values per K-row), not stride 152. The 4 bytes at k*8 + 8 to k*8 + 11 span K-rows k and k+1. This is the correct layout for the MMA instruction.

So the B fragment expects the data to be stored with stride 8 (N=8), and the 4 bytes in b_regs[0] can span up to 2 K-rows. The total B storage is K * 8 = 32 * 8 = 256 bytes.

In our kernel, we store weights with stride kWStrideN = 152. The B fragment reads with stride 8, not 152. So the reads are incorrect!

The fix: store W_fp8 with stride 8 (N=8) instead of stride 152. But our tile has kTileN = 128 N-rows, not 8. We can't use stride 8 for the full tile.

Alternative: restructure the SMEM to store B fragment data contiguously with stride 8. For each N-tile, store the 16 N-rows needed by the B fragment in a contiguous block with stride 8.

This is getting too complex. Let me try a simpler approach: just use the original row-major storage and the original B fragment read (which was verified to work with uniform weights). The issue with random activations must be elsewhere.

Let me revert all the changes and go back to the original code, then investigate the actual root cause:

Ran cd /home/oldzhu/sglang && git log --oneline -10 2>...
Let me revert to the clean state before all the diagnostic commits:

Executed terminal commands for git and Python scripts
Back to the baseline: Cases 1 and 3 pass, Cases 2 and 4 fail.

The problem is consistently with non-uniform activations. Let me now check one final hypothesis: maybe the issue is with the "=f" vs "+f" pattern. Let me check the current state of the asm:

Read w4a8_fp8_qmma.cu, lines 115 to 145

The code is at the baseline state with "=f" + store pattern. Let me try changing back to "+f" (which was the original before my changes):

Edited
w4a8_fp8_qmma.cu+5-7
Ran cd /home/oldzhu/sglang && git add sgl-kernel/csrc/...
Ran cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
Running cd /home/oldzhu/sglang && python3 scripts/fcloud/f...
cd /home/oldzhu/sglang && python3 scripts/fcloud/fcloud_exec.py exec --timeout 120 'nvcc -shared -o /root/submission_sim/libw4a8_fused_gemm.so /root/sglang-minicpm/sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu -O3 -arch=sm_120a --expt-relaxed-constexpr -std=c++17 -I /app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/include -I /app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I /app/.python-3.10/include/python3.10 -I /usr/local/cuda/include -L /app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/lib -lc10 -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -Xcompiler -fPIC && echo BUILD_OK 2>&1' 2>&1
 
Running
....
