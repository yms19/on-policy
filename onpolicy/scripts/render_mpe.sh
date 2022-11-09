#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=1
num_agents=5
num_good_agents=4
num_adversaries=1
num_bubbles=2
d_range=0.25
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=2 python render/render_mpe.py --save_gifs --share_policy \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_bubbles ${num_bubbles} --d_range ${d_range} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render \
    --episode_length 240 --render_episodes 1 --use_wandb
done
