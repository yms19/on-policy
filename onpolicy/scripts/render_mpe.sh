#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=0
num_agents=2
num_good_agents=2
num_adversaries=1
d_range=0.25
algo="rmappo"
exp="without_attention"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --use_ReLU --save_gifs \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} \
    --d_range ${d_range} --adversary_speed 0.0003 \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 2 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render \
    --episode_length 270 --script_length 70 --render_episodes 1 --use_wandb 
    # --model_dir "./results/MPE/simple_tag/rmappo/without_attention/wandb/run-20230510_070737-hse1sgg7/files"
done
