finally I submitted the two packages(both including sgl-kernel optimization) to offcial site - 
1. one is 32 calibration samples  with stratified sampling 
SOAR_GPTQ_CALIBRATION_SAMPLING:-stratified
SOAR_GPTQ_CALIBRATION_SAMPLES:-32
2. one is 8 clibration samples with sequential sampling
SOAR_GPTQ_CALIBRATION_SAMPLING = sequential
SOAR_GPTQ_CALIBRATION_SAMPLES = 8
and I got the below tow scores:
32 samples stratified
Score
{
  "acc": 96.08,
  "acc_ori": 76.87,
  "final_score": 0.0,
  "benchmark_duration": {
    "S1": 462.66,
    "S8": 661.83,
    "Smax": 1202.72
  }
}

8 samples sequential sampling
Score
{
  "acc": 92.56,
  "acc_ori": 74.04,
  "final_score": 0.0,
  "benchmark_duration": {
    "S1": 462.17,
    "S8": 661.55,
    "Smax": 1205.45
  }
}
comparing with the last 2 succss submissions, no improving on benchmark(even worser than 2nd time ) but accuracy dropped much <97%

the 2nd official score:(no sgl-kernel optimizarion )
 "acc_ori": 80.07,"S1": 458.78,"S8": 634.35,"Smax": 1140.66
the 3rd officl score:(including sgl-kernel optimization)
 "acc_ori": 78.4,"S1": 462.27,"S8": 663.66,"Smax": 1203.4

by the way, the 4th week champion blog posted already as the below:
https://mp.weixin.qq.com/s/MUYvxhb38XYxCd4ciaptYw
pls read and it can be used as one reference as our next actions to improce benchmark and benchmark.

I think the next:
1. continue to improve benchmark in all layer - from python to kernel to native ops.
2. continue to improve accuracy to 100%.
one thing I want to note is that the change hybrid_linear_attn_backend.py did not involve in the latest two submission. but I  tested in fcloud and seems no much improving. 
