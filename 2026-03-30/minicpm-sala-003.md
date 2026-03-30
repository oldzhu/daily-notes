I got this time officail scope as the below:
Score
{
  "acc": 97.33,
  "acc_ori": 77.87,
  "final_score": 54.24,
  "benchmark_duration": {
    "S1": 440.61,
    "S8": 626.43,
    "Smax": 1138.48
  }
}

comparing with the previous 3 official scores:
the 1st official scope 
"acc_ori": 80.67 "S1": 539.46,"S8": 747.59,"Smax": 1365.8

the 2nd official score:
 "acc_ori": 80.07,"S1": 458.78,"S8": 634.35,"Smax": 1140.66

the 3rd officl score:
 "acc_ori": 78.4,"S1": 462.27,"S8": 663.66,"Smax": 1203.4

 we could see although the benchmark speed improved at s1,s8 and smax all level, but due to acc_ori dropped a little bit so that  the final score is still lower than the best one of the previous.  Due to the accuracy is not the same each time,I will submit one time to see if can get better official score. 
 in our local side, I will test the change 0061 to see if it improving the benchmark and feedback the result. after the testing for 0061 dones,  let us continue :
 1. continue to improve accuracy - try best to reach 100% (acc_ori>=80%)
 2. continue to improve benchmark speed. 
