import sys
import os
import copy
import json
import time
import argparse

from grpc import server


command_list = [ 
    "nohup python train.py  --save-dir models/s9/rm_s9_e01/  --server-name FeiNewML --scenario simple_spread_multi_agent  --exp-name rm_s9_e01 > results/s9/rm_s9_e01.out 2> results/s9/rm_s9_e01.err &"
    ,
    "nohup python train.py  --save-dir models/s10/rm_s10_e01/  --server-name FeiNewML --scenario simple_adversary_multi_agent  --exp-name rm_s10_e01 > results/s10/rm_s10_e01.out 2> results/s10/rm_s10_e01.err &"
    ,
    "nohup python train.py  --save-dir models/s11/rm_s11_e01/  --server-name FeiNewML --scenario simple_push_multi_agent  --exp-name rm_s11_e01 > results/s11/rm_s11_e01.out 2> results/s11/rm_s11_e01.err &"
    ,
    "nohup python train.py  --save-dir models/s12/rm_s12_e01/  --server-name FeiNewML --scenario simple_tag_multi_agent  --exp-name rm_s12_e01 > results/s12/rm_s12_e01.out 2> results/s12/rm_s12_e01.err &"
                   ]

def run(sleep_time = 3600*2):
    for opt in command_list:
        print(opt)
        os.system(opt)
        print("------------------------sleep------------------------")
        time.sleep(sleep_time) # sleep for 5h

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning Experiments Excution")
    parser.add_argument("--server-name", type=str, default="FeiNewML", help="FeiML, FeiNewML, Miao_Exxact")
    parser.add_argument("--sleep-time", type=int, default=18000, help="3600 sec = 1 hour")
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    run(arglist.sleep_time)
