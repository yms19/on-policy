#!/bin/sh
env="MPE"
scenario="simple_tag"
num_landmarks=0
num_agents=4
num_good_agents=4
num_adversaries=1
d_range=0.25
algo="rmappo"
exp="way_points(noise)"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=3 python render/render_mpe.py --use_ReLU --activation_id 0 \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} \
    --d_range ${d_range}  --adversary_speed 0.0003 --p_noise 1 --detect_noise 0.15 \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 1 \
    --n_training_threads 1 --n_rollout_threads 1 --use_render \
    --episode_length 12 --world_length 240 --inference_interval 15 --render_episodes 100 --use_wandb \
    --model_dir "./results/MPE/simple_tag/rmappo/way_points/wandb/run-20230512_174909-ew1g90wo/files"
done
