I perform 2 times correctness evaluation for both non-quant model and quant model
non-quant
1st time:
verage Score: 79.58%
Total Duration: 3796.06 s
Total Tokens: In=8644166, Out=989665
Average Tokens/Sample: In=57627.8, Out=6597.8
Overall TPS (Output): 260.71 tokens/s
Per-task Accuracy
cwe: count=30 correct=25.7 accuracy=85.67% avg_in=74163.4 avg_out=17959.5
fwe: count=30 correct=29.6667 accuracy=98.89% avg_in=68153.7 avg_out=7194.3
mcq: count=30 correct=15.0 accuracy=50.00% avg_in=269.9 avg_out=7376.2
niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=356.4
qa: count=30 correct=19.0 accuracy=63.33% avg_in=71569.1 avg_out=102.4

Per-length-bucket Accuracy
len_0_4k: count=30 correct=15.0 accuracy=50.00% avg_in=269.9 avg_out=7376.2
len_32k_128k: count=80 correct=69.3 accuracy=86.62% avg_in=92784.4 avg_out=7778.0
len_4k_32k: count=40 correct=35.0667 accuracy=87.67% avg_in=30332.8 avg_out=3653.4

Per-task-length-bucket Accuracy
task=cwe|len_32k_128k: count=20 correct=16.3 accuracy=81.50% avg_in=95471.0 avg_out=23398.5
task=cwe|len_4k_32k: count=10 correct=9.4 accuracy=94.00% avg_in=31548.2 avg_out=7081.5
task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=7252.6
task=fwe|len_4k_32k: count=10 correct=9.6667 accuracy=96.67% avg_in=29697.8 avg_out=7077.9
task=mcq|len_0_4k: count=30 correct=15.0 accuracy=50.00% avg_in=269.9 avg_out=7376.2
task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=361.4
task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=346.4
task=qa|len_32k_128k: count=20 correct=13.0 accuracy=65.00% avg_in=92977.5 avg_out=99.6
task=qa|len_4k_32k: count=10 correct=6.0 accuracy=60.00% avg_in=28752.2 avg_out=107.9

2nd time:
Average Score: 83.02%
Total Duration: 3748.04 s
Total Tokens: In=8644166, Out=820989
Average Tokens/Sample: In=57627.8, Out=5473.3
Overall TPS (Output): 219.04 tokens/s

Per-task Accuracy
cwe: count=30 correct=27.2 accuracy=90.67% avg_in=74163.4 avg_out=13547.5
fwe: count=30 correct=29.3333 accuracy=97.78% avg_in=68153.7 avg_out=5059.9
mcq: count=30 correct=17.0 accuracy=56.67% avg_in=269.9 avg_out=8308.6
niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=347.9
qa: count=30 correct=21.0 accuracy=70.00% avg_in=71569.1 avg_out=102.4

Per-length-bucket Accuracy
len_0_4k: count=30 correct=17.0 accuracy=56.67% avg_in=269.9 avg_out=8308.6
len_32k_128k: count=80 correct=71.5 accuracy=89.38% avg_in=92784.4 avg_out=6103.2
len_4k_32k: count=40 correct=36.0333 accuracy=90.08% avg_in=30332.8 avg_out=2086.8

Per-task-length-bucket Accuracy
task=cwe|len_32k_128k: count=20 correct=17.5 accuracy=87.50% avg_in=95471.0 avg_out=16778.2
task=cwe|len_4k_32k: count=10 correct=9.7 accuracy=97.00% avg_in=31548.2 avg_out=7086.0
task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=7188.3
task=fwe|len_4k_32k: count=10 correct=9.3333 accuracy=93.33% avg_in=29697.8 avg_out=803.1
task=mcq|len_0_4k: count=30 correct=17.0 accuracy=56.67% avg_in=269.9 avg_out=8308.6
task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=345.2
task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=353.2
task=qa|len_32k_128k: count=20 correct=14.0 accuracy=70.00% avg_in=92977.5 avg_out=101.2
task=qa|len_4k_32k: count=10 correct=7.0 accuracy=70.00% avg_in=28752.2 avg_out=105.0

quant model:
1st time:
Average Score: 77.73%
Total Duration: 3216.68 s
Total Tokens: In=8644166, Out=1540198
Average Tokens/Sample: In=57627.8, Out=10268.0
Overall TPS (Output): 478.82 tokens/s

Per-task Accuracy
cwe: count=30 correct=23.6 accuracy=78.67% avg_in=74163.4 avg_out=26592.3
fwe: count=30 correct=30.0 accuracy=100.00% avg_in=68153.7 avg_out=15813.2
mcq: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=8469.2
niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=362.8
qa: count=30 correct=15.0 accuracy=50.00% avg_in=71569.1 avg_out=102.4

Per-length-bucket Accuracy
len_0_4k: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=8469.2
len_32k_128k: count=80 correct=64.9 accuracy=81.12% avg_in=92784.4 avg_out=13414.5
len_4k_32k: count=40 correct=33.7 accuracy=84.25% avg_in=30332.8 avg_out=5324.0

Per-task-length-bucket Accuracy
task=cwe|len_32k_128k: count=20 correct=14.9 accuracy=74.50% avg_in=95471.0 avg_out=33089.9
task=cwe|len_4k_32k: count=10 correct=8.7 accuracy=87.00% avg_in=31548.2 avg_out=13597.2
task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=20109.5
task=fwe|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=29697.8 avg_out=7220.6
task=mcq|len_0_4k: count=30 correct=18.0 accuracy=60.00% avg_in=269.9 avg_out=8469.2
task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=356.9
task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=374.6
task=qa|len_32k_128k: count=20 correct=10.0 accuracy=50.00% avg_in=92977.5 avg_out=101.8
task=qa|len_4k_32k: count=10 correct=5.0 accuracy=50.00% avg_in=28752.2 avg_out=103.5

2nd time:
Average Score: 76.67%
Total Duration: 3062.23 s
Total Tokens: In=8644166, Out=1293308
Average Tokens/Sample: In=57627.8, Out=8622.1
Overall TPS (Output): 422.34 tokens/s

Per-task Accuracy
cwe: count=30 correct=24.0 accuracy=80.00% avg_in=74163.4 avg_out=15843.3
fwe: count=30 correct=30.0 accuracy=100.00% avg_in=68153.7 avg_out=15845.9
mcq: count=30 correct=16.0 accuracy=53.33% avg_in=269.9 avg_out=10941.3
niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=376.9
qa: count=30 correct=15.0 accuracy=50.00% avg_in=71569.1 avg_out=102.9

Per-length-bucket Accuracy
len_0_4k: count=30 correct=16.0 accuracy=53.33% avg_in=269.9 avg_out=10941.3
len_32k_128k: count=80 correct=66.0 accuracy=82.50% avg_in=92784.4 avg_out=9378.5
len_4k_32k: count=40 correct=33.0 accuracy=82.50% avg_in=30332.8 avg_out=5369.7

Per-task-length-bucket Accuracy
task=cwe|len_32k_128k: count=20 correct=16.0 accuracy=80.00% avg_in=95471.0 avg_out=16843.7
task=cwe|len_4k_32k: count=10 correct=8.0 accuracy=80.00% avg_in=31548.2 avg_out=13842.4
task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=20210.6
task=fwe|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=29697.8 avg_out=7116.4
task=mcq|len_0_4k: count=30 correct=16.0 accuracy=53.33% avg_in=269.9 avg_out=10941.3
task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=360.9
task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=409.0
task=qa|len_32k_128k: count=20 correct=10.0 accuracy=50.00% avg_in=92977.5 avg_out=99.0
task=qa|len_4k_32k: count=10 correct=5.0 accuracy=50.00% avg_in=28752.2 avg_out=110.9

Could u compare and give the list of the task and task-length in the order of accuracy loss.
