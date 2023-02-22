#!/bin/sh
env="MPE"
scenario="simple_tag"  # simple_speaker_listener # simple_reference
num_landmarks=0
num_agents=4
num_good_agents=4
num_adversaries=1
d_range=0.25
algo="rmappo"
exp="escape_nearest"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_mpe.py --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed 1 \
    --num_good_agents ${num_good_agents} --num_adversaries ${num_adversaries} --d_range ${d_range} \
    --n_training_threads 1 --n_rollout_threads 256 --num_mini_batch 1 --episode_length 160 --script_length 80 \
    --num_env_steps 100000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --user_name "yangxt19" --wandb_name "yangxt19" 
    # --model_dir "./results/MPE/simple_tag/rmappo/escape_nearest/wandb/run-20230220_160716-20kp2kk2/files" 
done