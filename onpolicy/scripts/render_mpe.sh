#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=0
num_agents=4
num_good_agents=4
num_adversaries=1
num_bubbles=0
d_range=0.25
algo="rmappo"
exp="escape_nearest"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=3 python render/render_mpe.py --use_ReLU --save_gifs \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_bubbles ${num_bubbles} --d_range ${d_range} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 1 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render \
    --episode_length 240 --script_length 0 --render_episodes 1 --use_wandb 
    # --model_dir "./results/MPE/simple_tag/rmappo/with_detect(mask)/wandb/run-20230215_072425-1hkw3wgd/files"
done
