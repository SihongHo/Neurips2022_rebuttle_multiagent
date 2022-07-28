# Neurips2022_rebuttle_multiagent

## Task

1. Run baseline algorithms (MADDPG and M3DDPG) on MPE (with more agents) and get policies. [We first finish this step]

2. Testing policies under the optimal adversaries obtained from RMAAC.

3. Computing statistics for testing data.

## 3 Main Resource

### MADDPG 

resource: https://github.com/SihongHo/Baseline_Project_MADDPG

Feel free to use another resource if it's more convenient

Please use these commands to train ma models:

``nohup python train.py  --save-dir models/s9/ma_s9_e01/    --scenario simple_spread_multi_agent  --exp-name ma_s9_e01 > results/s9/ma_s9_e01.out 2> results/s9/ma_s9_e01.err &``,

``nohup python train.py  --save-dir models/s10/ma_s10_e01/    --scenario simple_adversary_multi_agent  --exp-name ma_s10_e01 > results/s10/ma_s10_e01.out 2> results/s10/ma_s10_e01.err &``,

``nohup python train.py  --save-dir models/s11/ma_s11_e01/    --scenario simple_push_multi_agent  --exp-name ma_s11_e01 > results/s11/ma_s11_e01.out 2> results/s11/ma_s11_e01.err &``,

``nohup python train.py  --save-dir models/s12/ma_s12_e01/    --scenario simple_tag_multi_agent  --exp-name ma_s12_e01 > results/s12/ma_s12_e01.out 2> results/s12/ma_s12_e01.err &``

### M3DDPG

resource: https://github.com/SihongHo/Baseline_Project_M3DDPG

Feel free to use another resource if it's more convenient

Please use these commands to train ma models:

``nohup python train.py  --save-dir models/s9/m3_s9_e01/    --scenario simple_spread_multi_agent  --exp-name m3_s9_e01 > results/s9/m3_s9_e01.out 2> results/s9/m3_s9_e01.err &``,

``nohup python train.py  --save-dir models/s10/m3_s10_e01/    --scenario simple_adversary_multi_agent  --exp-name m3_s10_e01 > results/s10/m3_s10_e01.out 2> results/s10/m3_s10_e01.err &``,

``nohup python train.py  --save-dir models/s11/m3_s11_e01/    --scenario simple_push_multi_agent  --exp-name m3_s11_e01 > results/s11/m3_s11_e01.out 2> results/s11/m3_s11_e01.err &``,

 ``nohup python train.py  --save-dir models/s12/m3_s12_e01/    --scenario simple_tag_multi_agent  --exp-name m3_s12_e01 > results/s12/m3_s12_e01.out 2> results/s12/m3_s12_e01.err &``
                   

### Modified MPE

Please reinstall the MEP using the following resource.

resource: https://github.com/SihongHo/multiagent-particle-envs

what have added: 4 scenarios with more agents

1. multiagent/scenarios/simple_adversary_multi_agent.py

2. multiagent/scenarios/simple_push_multi_agent.py

3. multiagent/scenarios/simple_spread_multi_agent.py

4. multiagent/scenarios/simple_tag_multi_agent.py

## Runinng on Servers

1. On all servers, I have configured the build environment. You can use ``conda activate hsh_maddpg``. 

2. If hsh_maddpg does not work, you can try create a new one by using:

``conda create -n hsh_maddpg python=3.5.4``

``conda activate hsh_maddpg``

``conda install numpy=1.14.5``

``# conda install -c anaconda tensorflow-gpu``

``conda install tensorflow``

``# conda install gym=0.10.5``

``pip install gym==0.10.5``


## Github Cooperate Note

1. Please keep the main branch unchanged. Unless I merge some branches into the main branch.

2. Create new branches to run each task. For example branch_ma, brach_m3, etc.

## Summary of MPE

https://www.notion.so/benchmark-parameters-90e268adf8044248b1ca91c780553f18