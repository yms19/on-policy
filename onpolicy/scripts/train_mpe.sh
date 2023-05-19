#!/bin/sh
env="MPE"
scenario="simple_tag"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
num_good_agents=4
num_adversaries=1
d_range=0.25
algo="rmappo"
exp="self_play(possibility)"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_mpe.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 1 --save_history_interval 200 \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} \
    --d_range ${d_range} --adversary_speed 0.0003 \
    --n_training_threads 1 --n_rollout_threads 256 --num_mini_batch 1 --episode_length 170 --script_length 70 \
    --num_env_steps 100000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --user_name "yangxt19" --wandb_name "yangxt19" 
    # --model_dir "./results/MPE/simple_tag/rmappo/without_attention/wandb/run-20230510_070737-hse1sgg7/files"
done