#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=0
num_agents=5
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
    CUDA_VISIBLE_DEVICES=3 python ./render_mpe.py --use_ReLU --save_gifs \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --num_bubbles ${num_bubbles} --d_range ${d_range} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 100 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render \
    --episode_length 180 --render_episodes 1 --use_wandb \
    # --model_dir "./results/MPE/simple_tag/rmappo/with_detect/wandb/run-20221219_102117-3oa3pxmb/files"
done
