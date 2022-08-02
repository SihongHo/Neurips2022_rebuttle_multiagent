import sys
import os
import copy
import json
import time

command_list_optimal =[
"nohup python train_with_adv.py --num-adversaries 0 --load-dir models/s1/m3_s1_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_reference --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s1/ma_s1_e01/ --exp-name eva_m3_s1_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark > results/s1/eva_m3_s1_e01_Optimal.out 2> results/s1/eva_m3_s1_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 0 --load-dir models/s2/m3_s2_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_speaker_listener --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s2/ma_s2_e01/ --exp-name eva_m3_s2_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s2/eva_m3_s2_e01_Optimal.out 2> results/s2/eva_m3_s2_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 0 --load-dir models/s3/m3_s3_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_spread --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s3/ma_s3_e01/ --exp-name eva_m3_s3_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s3/eva_m3_s3_e01_Optimal.out 2> results/s3/eva_m3_s3_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 1 --load-dir models/s4/m3_s4_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_adversary --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s4/ma_s4_e01/ --exp-name eva_m3_s4_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s4/eva_m3_s4_e01_Optimal.out 2> results/s4/eva_m3_s4_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 1 --load-dir models/s5/m3_s5_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_crypto --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s5/ma_s5_e01/ --exp-name eva_m3_s5_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s5/eva_m3_s5_e01_Optimal.out 2> results/s5/eva_m3_s5_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 1 --load-dir models/s6/m3_s6_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_push --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s6/ma_s6_e01/ --exp-name eva_m3_s6_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s6/eva_m3_s6_e01_Optimal.out 2> results/s6/eva_m3_s6_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 3 --load-dir models/s7/m3_s7_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_tag --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s7/ma_s7_e01/ --exp-name eva_m3_s7_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s7/eva_m3_s7_e01_Optimal.out 2> results/s7/eva_m3_s7_e01_Optimal.err &"
,
"nohup python train_with_adv.py --num-adversaries 4 --load-dir models/s8/m3_s8_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_world_comm --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s8/ma_s8_e01/ --exp-name eva_m3_s8_e01_Optimal --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s8/eva_m3_s8_e01_Optimal.out 2> results/s8/eva_m3_s8_e01_Optimal.err &"
]

command_list_no =[
"nohup python test_without_noise.py --num-adversaries 0 --load-dir models/s1/m3_s1_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_reference --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s1/ma_s1_e01/ --exp-name eva_m3_s1_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark > results/s1/eva_m3_s1_e01_No.out 2> results/s1/eva_m3_s1_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 0 --load-dir models/s2/m3_s2_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_speaker_listener --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s2/ma_s2_e01/ --exp-name eva_m3_s2_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s2/eva_m3_s2_e01_No.out 2> results/s2/eva_m3_s2_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 0 --load-dir models/s3/m3_s3_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_spread --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s3/ma_s3_e01/ --exp-name eva_m3_s3_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s3/eva_m3_s3_e01_No.out 2> results/s3/eva_m3_s3_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 1 --load-dir models/s4/m3_s4_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_adversary --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s4/ma_s4_e01/ --exp-name eva_m3_s4_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s4/eva_m3_s4_e01_No.out 2> results/s4/eva_m3_s4_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 1 --load-dir models/s5/m3_s5_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_crypto --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s5/ma_s5_e01/ --exp-name eva_m3_s5_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s5/eva_m3_s5_e01_No.out 2> results/s5/eva_m3_s5_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 1 --load-dir models/s6/m3_s6_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_push --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s6/ma_s6_e01/ --exp-name eva_m3_s6_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s6/eva_m3_s6_e01_No.out 2> results/s6/eva_m3_s6_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 3 --load-dir models/s7/m3_s7_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_tag --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s7/ma_s7_e01/ --exp-name eva_m3_s7_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s7/eva_m3_s7_e01_No.out 2> results/s7/eva_m3_s7_e01_No.err &"
,
"nohup python test_without_noise.py --num-adversaries 4 --load-dir models/s8/m3_s8_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_world_comm --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s8/ma_s8_e01/ --exp-name eva_m3_s8_e01_No --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s8/eva_m3_s8_e01_No.out 2> results/s8/eva_m3_s8_e01_No.err &"
]

command_list_norm =[
"nohup python test_with_norm.py --num-adversaries 0 --load-dir models/s1/m3_s1_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_reference --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s1/ma_s1_e01/ --exp-name eva_m3_s1_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark > results/s1/eva_m3_s1_e01_Random.out 2> results/s1/eva_m3_s1_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 0 --load-dir models/s2/m3_s2_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_speaker_listener --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s2/ma_s2_e01/ --exp-name eva_m3_s2_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s2/eva_m3_s2_e01_Random.out 2> results/s2/eva_m3_s2_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 0 --load-dir models/s3/m3_s3_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_spread --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s3/ma_s3_e01/ --exp-name eva_m3_s3_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s3/eva_m3_s3_e01_Random.out 2> results/s3/eva_m3_s3_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 1 --load-dir models/s4/m3_s4_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_adversary --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s4/ma_s4_e01/ --exp-name eva_m3_s4_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s4/eva_m3_s4_e01_Random.out 2> results/s4/eva_m3_s4_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 1 --load-dir models/s5/m3_s5_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_crypto --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s5/ma_s5_e01/ --exp-name eva_m3_s5_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s5/eva_m3_s5_e01_Random.out 2> results/s5/eva_m3_s5_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 1 --load-dir models/s6/m3_s6_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_push --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s6/ma_s6_e01/ --exp-name eva_m3_s6_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s6/eva_m3_s6_e01_Random.out 2> results/s6/eva_m3_s6_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 3 --load-dir models/s7/m3_s7_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_tag --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s7/ma_s7_e01/ --exp-name eva_m3_s7_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s7/eva_m3_s7_e01_Random.out 2> results/s7/eva_m3_s7_e01_Random.err &"
,
"nohup python test_with_norm.py --num-adversaries 4 --load-dir models/s8/m3_s8_mmmddpg_mmmddpg_e20/model-59000 --scenario simple_world_comm --adv-dir ../../MyAlgorithm/v4-rmaddpg-master/experiments/models/s8/ma_s8_e01/ --exp-name eva_m3_s8_e01_Random --good-policy mmmddpg --bad-policy mmmddpg --benchmark  > results/s8/eva_m3_s8_e01_Random.out 2> results/s8/eva_m3_s8_e01_Random.err &"
]

# for command in command_list_optimal:
# 	print(command)
# 	os.system(command)
# 	print("------------------------sleep------------------------")
# 	time.sleep(600) # sleep for 10min

for command in command_list_no:
	print(command)
	os.system(command)
	print("------------------------sleep------------------------")
	time.sleep(600) # sleep for 10min

for command in command_list_norm:
    print(command)
    os.system(command)
    print("------------------------sleep------------------------")
    time.sleep(600) # sleep for 10min
