I tested two times with change 0061 and got the below:
"accuracy": 76.33% "s1 duration": 117.52424013055861,"s8 duration": 25.50426913611591,"smax duration": 20.771070692688227
"accuracy": 78.98% "s1 duration": 118.05890387669206,"s8 duration": 25.502217521890998,"smax duration": 20.779481925070286
comparing with the last time withour 0061, 
"accuracy": 78.64% "s1 duration": 117.49830645509064,"s8 duration": 25.51103823632002,"smax duration": 20.773862736299634
we could see Benchmark has no much improvement and the accurace is simialr as before which is not stable - one time even below <77%. 
for the detailed accuracy distribution:
1st time:
verage Score: 76.33%
Total Duration: 2853.91 s
Total Tokens: In=8644166, Out=1269997
Average Tokens/Sample: In=57627.8, Out=8466.6
Overall TPS (Output): 445.00 tokens/s

Per-task Accuracy
  cwe: count=30 correct=24.5 accuracy=81.67% avg_in=74163.4 avg_out=20087.8
  fwe: count=30 correct=30.0 accuracy=100.00% avg_in=68153.7 avg_out=7204.3
  mcq: count=30 correct=13.0 accuracy=43.33% avg_in=269.9 avg_out=14587.3
  niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=350.5
  qa: count=30 correct=17.0 accuracy=56.67% avg_in=71569.1 avg_out=103.3

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=13.0 accuracy=43.33% avg_in=269.9 avg_out=14587.3
  len_32k_128k: count=80 correct=67.5 accuracy=84.38% avg_in=92784.4 avg_out=8573.8
  len_4k_32k: count=40 correct=34.0 accuracy=85.00% avg_in=30332.8 avg_out=3661.9

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=15.5 accuracy=77.50% avg_in=95471.0 avg_out=23365.5
  task=cwe|len_4k_32k: count=10 correct=9.0 accuracy=90.00% avg_in=31548.2 avg_out=13532.4
  task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=10485.8
  task=fwe|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=29697.8 avg_out=641.4
  task=mcq|len_0_4k: count=30 correct=13.0 accuracy=43.33% avg_in=269.9 avg_out=14587.3
  task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=341.1
  task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=369.4
  task=qa|len_32k_128k: count=20 correct=12.0 accuracy=60.00% avg_in=92977.5 avg_out=102.8
  task=qa|len_4k_32k: count=10 correct=5.0 accuracy=50.00% avg_in=28752.2 avg_out=104.3
2nd time:
Average Score: 78.98%
Total Duration: 2761.60 s
Total Tokens: In=8644166, Out=1127564
Average Tokens/Sample: In=57627.8, Out=7517.1
Overall TPS (Output): 408.30 tokens/s

Per-task Accuracy
  cwe: count=30 correct=25.8 accuracy=86.00% avg_in=74163.4 avg_out=11434.8
  fwe: count=30 correct=29.6667 accuracy=98.89% avg_in=68153.7 avg_out=11525.3
  mcq: count=30 correct=16.0 accuracy=53.33% avg_in=269.9 avg_out=14164.6
  niah: count=30 correct=30.0 accuracy=100.00% avg_in=73982.7 avg_out=355.4
  qa: count=30 correct=17.0 accuracy=56.67% avg_in=71569.1 avg_out=105.3

Per-length-bucket Accuracy
  len_0_4k: count=30 correct=16.0 accuracy=53.33% avg_in=269.9 avg_out=14164.6
  len_32k_128k: count=80 correct=68.6 accuracy=85.75% avg_in=92784.4 avg_out=6942.4
  len_4k_32k: count=40 correct=33.8667 accuracy=84.67% avg_in=30332.8 avg_out=3680.9

Per-task-length-bucket Accuracy
  task=cwe|len_32k_128k: count=20 correct=16.6 accuracy=83.00% avg_in=95471.0 avg_out=13602.9
  task=cwe|len_4k_32k: count=10 correct=9.2 accuracy=92.00% avg_in=31548.2 avg_out=7098.8
  task=fwe|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=87381.7 avg_out=13704.5
  task=fwe|len_4k_32k: count=10 correct=9.6667 accuracy=96.67% avg_in=29697.8 avg_out=7166.8
  task=mcq|len_0_4k: count=30 correct=16.0 accuracy=53.33% avg_in=269.9 avg_out=14164.6
  task=niah|len_32k_128k: count=20 correct=20.0 accuracy=100.00% avg_in=95307.5 avg_out=355.6
  task=niah|len_4k_32k: count=10 correct=10.0 accuracy=100.00% avg_in=31333.2 avg_out=355.0
  task=qa|len_32k_128k: count=20 correct=12.0 accuracy=60.00% avg_in=92977.5 avg_out=106.3
  task=qa|len_4k_32k: count=10 correct=5.0 accuracy=50.00% avg_in=28752.2 avg_out=103.1
comparing with one time high accuracy early as the below:
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
we could see the main dropped unstable categories are the 3 - qa,mcq,cwe
could we increase  the samples to 48 for calibration?(it may cause OOM(64 samples give OOM each time) - in case OOM, we can still use 32 with qa,mcq,cwe or try to fix OOM). what do u think?
=========
