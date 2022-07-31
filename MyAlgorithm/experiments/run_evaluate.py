import sys
import os
import copy
import json
import time
import argparse
from grpc import server

# nohup python run_evaluate.py > 2022_05_01_eva_run.out 2> 2022_05_01_eva_run.err &
# nohup python run_evaluate.py > 2022_05_02_eva_run.out 2> 2022_05_02_eva_run.err &
def get_command(scenario_index = 8, exp_index = 0, cuda_index = 1, alg = "ma", server_name = "FeiML", noise_type = "Optimal"):
    scenario_list = ["simple","simple_reference", "simple_speaker_listener", "simple_spread",
                        "simple_adversary", "simple_crypto", "simple_push",
                        "simple_tag", "simple_world_comm"]
    exp_name_list = ["e0" + str(i) for i in range(1,10,1)] + ["e" + str(i) for i in range(10,21,1)]
    model_name = alg + "_s" + str(scenario_index) + "_" + str(exp_name_list[exp_index])
    exp_name = "eva_" + model_name + "_" + noise_type
    exp_path = " > results/s" + str(scenario_index) + "/" + exp_name + ".out 2> results/s" \
                    + str(scenario_index) + "/" + exp_name + ".err &"

    opt1 = dict()
    opt1['server-name'] = server_name
    opt1['scenario'] = scenario_list[scenario_index]
    opt1['save-dir'] = "models/s" + str(scenario_index) + "/" + model_name + "/"
    opt1['benchmark'] = ''
    opt1['perturb'] = noise_type
    opt2 = dict()
    opt2['exp-name'] = exp_name + exp_path

    def generate_command(opt1, opt2, cuda_index):
        cmd = 'CUDA_VISIBLE_DEVICES=' + str(cuda_index) + ' nohup python evaluate.py'
        for opt, val in opt1.items():
            cmd += ' --' + opt + ' ' + str(val)
        for opt, val in opt2.items():
            cmd += ' --' + opt + ' ' + str(val)
        return cmd
    
    return generate_command(opt1, opt2, cuda_index)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning Experiments Excution")
    parser.add_argument("--server-name", type=str, default="FeiML", help="FeiML, FeiNewML, Miao_Exxact")
    parser.add_argument("--exp-index", type=int, default=0, help="FeiML, FeiNewML, Miao_Exxact = 1 2 3")
    return parser.parse_args()

def run(scenario_index, exp_index, server_name, noise_type):
    opt = get_command(scenario_index = scenario_index, exp_index = exp_index, cuda_index = 1, alg = "ma", server_name = server_name, noise_type = noise_type)
    conda_command = "conda activate hsh_maddpg"
    # print(conda_command)
    print(opt)
    # os.system(conda_command)
    os.system(opt)
    print("------------------------sleep------------------------")
    time.sleep(300) # sleep for 5min


if __name__ == '__main__':
    for i in range(8,9):
        run(scenario_index = i, server_name = "server", exp_index = 0, noise_type = "No")
    for i in range(8,9):
        run(scenario_index = i, server_name = "server", exp_index = 0, noise_type = "Optimal")
    for i in range(1,9):
        run(scenario_index = i, server_name = "server", exp_index = 0, noise_type = "Random")