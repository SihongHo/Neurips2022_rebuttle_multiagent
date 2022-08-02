import sys
import os
import copy
import json
import time

command_list_optimal =[
"nohup python train_with_adv.py --load-dir models/s10/ma_s10_e01/ --scenario simple_adversary_multi_agent --adv-dir ../../MyAlgorithm/experiments/models/s10/rm_s10_e01/ --exp-name eva_ma_s10_e01_Optimal --benchmark > results/s10/eva_ma_s10_e01_Optimal.out 2> results/s10/eva_ma_s10_e01_Optimal.err &",
"nohup python train_with_adv.py --load-dir models/s11/ma_s11_e01/ --scenario  simple_push_multi_agent --adv-dir ../../MyAlgorithm/experiments/models/s11/rm_s11_e01/ --exp-name eva_ma_s11_e01_Optimal --benchmark > results/s11/eva_ma_s11_e01_Optimal.out 2> results/s11/eva_ma_s11_e01_Optimal.err &",
"nohup python train_with_adv.py --load-dir models/s12/ma_s12_e01/ --scenario  simple_tag_multi_agent --adv-dir ../../MyAlgorithm/experiments/models/s12/rm_s12_e01/ --exp-name eva_ma_s12_e01_Optimal --benchmark > results/s12/eva_ma_s12_e01_Optimal.out 2> results/s12/eva_ma_s12_e01_Optimal.err &"    
]
#nohup python train_with_adv.py --num-adversaries 0 --load-dir models/s9/ma_s9_e01/ --scenario simple_spread_multi_agent --adv-dir ../../MyAlgorithm/experiments/models/s9/rm_s9_e01/ --exp-name eva_ma_s9_e01_Optimal --benchmark > results/s9/eva_ma_s9_e01_Optimal.out 2> results/s9/eva_ma_s9_e01_Optimal.err &

command_list_no =[
"nohup python train.py --load-dir models/s10/ma_s10_e01/ --scenario simple_adversary_multi_agent --exp-name eva_ma_s10_e01_No --benchmark > results/s10/eva_ma_s10_e01_No.out 2> results/s10/eva_ma_s10_e01_No.err &"
,
"nohup python train.py --load-dir models/s11/ma_s11_e01/ --scenario  simple_push_multi_agent --exp-name eva_ma_s11_e01_No --benchmark > results/s11/eva_ma_s11_e01_No.out 2> results/s11/eva_ma_s11_e01_No.err &"
,
"nohup python train.py --load-dir models/s12/ma_s12_e01/ --scenario  simple_tag_multi_agent --exp-name eva_ma_s12_e01_No --benchmark > results/s12/eva_ma_s12_e01_No.out 2> results/s12/eva_ma_s12_e01_No.err &"
]

command_list_random =[
"nohup python test_with_norm.py --load-dir models/s10/ma_s10_e01/ --scenario simple_adversary_multi_agent --exp-name eva_ma_s10_e01_Random --benchmark > results/s10/eva_ma_s10_e01_Random.out 2> results/s10/eva_ma_s10_e01_Random.err &"
,
"nohup python test_with_norm.py --load-dir models/s11/ma_s11_e01/ --scenario  simple_push_multi_agent --exp-name eva_ma_s11_e01_Random --benchmark > results/s11/eva_ma_s11_e01_Random.out 2> results/s11/eva_ma_s11_e01_Random.err &"
,
"nohup python test_with_norm.py --load-dir models/s12/ma_s12_e01/ --scenario  simple_tag_multi_agent --exp-name eva_ma_s12_e01_Random --benchmark > results/s12/eva_ma_s12_e01_Random.out 2> results/s12/eva_ma_s12_e01_Random.err &"
]

#for marl
command_list_no_rm = [
"nohup python test_without_noise.py --load-dir models/s10/rm_s10_e01/ --scenario simple_adversary_multi_agent --exp-name eva_rm_s10_e01_No --benchmark > results/s10/eva_rm_s10_e01_No.out 2> results/s10/eva_rm_s10_e01_No.err &",
"nohup python test_without_noise.py --load-dir models/s11/rm_s11_e01/ --scenario simple_push_multi_agent --exp-name eva_rm_s11_e01_No --benchmark > results/s11/eva_rm_s11_e01_No.out 2> results/s11/eva_rm_s11_e01_No.err &",
"nohup python test_without_noise.py --load-dir models/s12/rm_s12_e01/ --scenario simple_tag_multi_agent --exp-name eva_rm_s12_e01_No --benchmark > results/s12/eva_rm_s12_e01_No.out 2> results/s12/eva_rm_s12_e01_No.err &"
]

command_list_random_rm = [
"nohup python test_with_norm.py --load-dir models/s10/rm_s10_e01/ --scenario simple_adversary_multi_agent --exp-name eva_rm_s10_e01_Random --benchmark > results/s10/eva_rm_s10_e01_Random.out 2> results/s10/eva_rm_s10_e01_Random.err &",
"nohup python test_with_norm.py --load-dir models/s11/rm_s11_e01/ --scenario simple_push_multi_agent --exp-name eva_rm_s11_e01_Random --benchmark > results/s11/eva_rm_s11_e01_Random.out 2> results/s11/eva_rm_s11_e01_Random.err &",
"nohup python test_with_norm.py --load-dir models/s12/rm_s12_e01/ --scenario simple_tag_multi_agent --exp-name eva_rm_s12_e01_Random --benchmark > results/s12/eva_rm_s12_e01_Random.out 2> results/s12/eva_rm_s12_e01_Random.err &"
]
#"nohup python test_with_norm.py --load-dir models/s9/rm_s9_e01/ --scenario simple_spread_multi_agent --exp-name eva_rm_s9_e01_Random --benchmark > results/s9/eva_rm_s9_e01_Random.out 2> results/s9/eva_rm_s9_e01_Random.err &",

def func():
    for command in command_list_no:
        print(command)
        os.system(command)
        print("------------------------sleep------------------------")
        time.sleep(1200) # sleep for 20min

    for command in command_list_random:
        print(command)
        os.system(command)
        print("------------------------sleep------------------------")
        time.sleep(1200) # sleep for 20min
    
print(os.getcwd())

os.chdir("../../MyAlgorithm/experiments/")

print(os.getcwd())

def func1():
    for command in command_list_no_rm:
        print(command)
        os.system(command)
        print("------------------------sleep------------------------")
        time.sleep(1200) # sleep for 20min

for command in command_list_random_rm:
    print(command)
    os.system(command)
    print("------------------------sleep------------------------")
    time.sleep(1200) # sleep for 20min
