I tested two times with 90 samples qa,mcq,cwe + o_proj_w8 and got the below:
Average Score: 79.07%
Total Duration: 2796.07 s
Total Tokens: In=8644166, Out=1167222
Average Tokens/Sample: In=57627.8, Out=7781.5
Overall TPS (Output): 417.45 tokens/s

Per-task Accuracy
  cwe: count=30 correct=24.6 accuracy=82.00% avg_in=74163.4 avg_out=20267.9
  fwe: count=30 correct=30.0 accuracy=100.00% avg_in=68153.7 avg_out=5159.6
  mcq: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=12779.2
  niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=591.4
  qa: count=30 correct=16.0 accuracy=53.33% avg_in=71569.1 avg_out=109.3

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=12779.2
  len_32k_128k: count=80 correct=68.6 accuracy=85.75% avg_in=92784.4 avg_out=7006.8
  len_4k_32k: count=40 correct=32.0 accuracy=80.00% avg_in=30332.8 avg_out=5582.6

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=15.6 accuracy=78.00% avg_in=95471.0 avg_out=20241.0
  task=cwe|len_4k_32k: count=10 correct=9.0 accuracy=90.00% avg_in=31548.2 avg_out=20321.7
  task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=7329.3
  task=fwe|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=29697.8 avg_out=820.2
  task=mcq|len_0_4k: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=12779.2
  task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=355.1
  task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=1064.0
  task=qa|len_32k_128k: count=20 correct=13.0 accuracy=65.00% avg_in=92977.5 avg_out=101.7
  task=qa|len_4k_32k: count=10 correct=3.0 accuracy=30.00% avg_in=28752.2 avg_out=124.5

Average Score: 78.53%
Total Duration: 2854.52 s
Total Tokens: In=8644166, Out=1227781
Average Tokens/Sample: In=57627.8, Out=8185.2
Overall TPS (Output): 430.12 tokens/s

Per-task Accuracy
  cwe: count=30 correct=24.8 accuracy=82.67% avg_in=74163.4 avg_out=20064.9
  fwe: count=30 correct=30.0 accuracy=100.00% avg_in=68153.7 avg_out=7262.8
  mcq: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=13137.8
  niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=360.2
  qa: count=30 correct=15.0 accuracy=50.00% avg_in=71569.1 avg_out=100.3

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=13137.8
  len_32k_128k: count=80 correct=66.5 accuracy=83.12% avg_in=92784.4 avg_out=6937.4
  len_4k_32k: count=40 correct=33.3 accuracy=83.25% avg_in=30332.8 avg_out=6966.4

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=15.5 accuracy=77.50% avg_in=95471.0 avg_out=20044.4
  task=cwe|len_4k_32k: count=10 correct=9.3 accuracy=93.00% avg_in=31548.2 avg_out=20106.0
  task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=7232.1
  task=fwe|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=29697.8 avg_out=7324.1
  task=mcq|len_0_4k: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=13137.8
  task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=374.4
  task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=331.8
  task=qa|len_32k_128k: count=20 correct=11.0 accuracy=55.00% avg_in=92977.5 avg_out=98.7
  task=qa|len_4k_32k: count=10 correct=4.0 accuracy=40.00% avg_in=28752.2 avg_out=103.6

I will test another set 32 qa+cwe samples + o_proj_w8 to see if can get accuracy back to 80%.
meanwhilel, another other thing can be using w8 to make the accuracy better in case o_proj_w8 alone can not make accuracy back to 80%?
==========
Yes. The current result gives a fairly clear signal:

o_proj_w8 is not useless, but it is not the right full answer by itself.
It consistently helps mcq and slightly helps cwe, but it hurts qa badly, especially in the 4k-32k bucket. So the module is improving post-attention projection quality for short-form choice behavior, while not fixing, and maybe worsening, the routing or representation quality that qa depends on.

That means the next W8 targets should be ones that are more qa-sensitive than o_proj.

My recommendation order is:

1. qkv together at W8, but only for a narrow layer subset
Reason:
The qa drop looks more like attention-routing sensitivity than output projection sensitivity.
Important constraint:
Do not split q, k, and v precision inside one fused group. If we touch them, we should promote q_proj, k_proj, and v_proj together for the same layers.
Risk:
Higher speed cost than o_proj_w8, especially if applied globally.
Best scope:
Last 6 to 8 layers first, or only the sparse/full-attention layers if we can isolate them cleanly.

2. down_proj at W8
Reason:
This is a safer non-fused target than qkv and may help answer stability without touching the fused attention boundary.
Expected tradeoff:
Usually lower structural risk than qkv_w8, and likely smaller speed hit than promoting full qkv.

3. o_proj_w8 plus down_proj_w8, but only in the last layers
Reason:
Your current result suggests o_proj helps mcq. If we want to keep that benefit while repairing qa, combining it with a narrow down_proj upgrade in the last layers is a reasonable next step.

What I would avoid for now:

1. k-only, q-only, or v-only W8
Reason:
That is exactly the fused-boundary class of idea that already caused loader/format trouble before.

2. gate_proj or up_proj at W8 as the next move
Reason:
They are heavy MLP paths and are more likely to cost speed before giving a clear qa win.

3. 3-bit mixed MLP on the current gptq_marlin route
Reason:
The current marlin path here is really a 4-bit or 8-bit path, so 3-bit is not the next practical experiment.

The other important signal in your result is mcq output length. mcq is now at around 12k to 13k output tokens on average, which is extremely high for a short multiple-choice task. That means part of the remaining score problem may not be only weight precision. There is likely also a generation-behavior issue. So even if we keep pushing W8, I would treat W8 as a partial recovery tool, not assume it alone will get you back to 80.

So while you test 32 qa+cwe plus o_proj_w8, my current judgment is:

If that reaches about 80, keep the feature narrow and stop there.
If it still stays below 80, the next best W8 experiment is not more calibration. It is:
qkv_w8 on a narrow layer subset, or down_proj_w8 as the lower-risk alternative.

Concrete next-step recommendation:
1. Finish the 32 qa+cwe plus o_proj_w8 run.
2. If it still misses 80, I recommend the next approved feature be W8 qkv on a narrow layer subset, not global.
3. If you want the safer first variant before touching qkv, do down_proj_w8 instead.

If you want, I can prepare the next formal proposal as one feature:
CHANGE_0066: GPTQ mixed precision with subset qkv_w8
or
CHANGE_0066: GPTQ mixed precision with down_proj_w8

My recommendation between those two is:
subset qkv_w8 if you want the highest chance to recover qa,
down_proj_w8 if you want the lower implementation and runtime risk.
