#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=0
num_agents=4
num_good_agents=4
num_adversaries=1
num_bubbles=2
d_range=0.25
algo="rmappo"
exp="scripts"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=3 python render/render_mpe.py --use_ReLU \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_bubbles ${num_bubbles} --d_range ${d_range} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 1 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render \
    --episode_length 120 --script_length 120 --render_episodes 50 --use_wandb \
    # --model_dir "./results/MPE/simple_tag/rmappo/with_detect(mask)/wandb/run-20230209_100012-2cx1b1yd/files"
    # --model_dir "./results/MPE/simple_tag/rmappo/with_detect(mask)/wandb/run-20230209_100012-2cx1b1yd/files"
    # --model_dir "./results/MPE/simple_tag/rmappo/with_detect(mask)/wandb/run-20230209_100324-3isjj4zm/files"    
done
