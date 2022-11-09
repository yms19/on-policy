    
from random import random, randint
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from PIL import Image, ImageDraw, ImageFont

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self, display):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            infos = [[{'individual_reward': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_adversary': False}]]
            if self.all_args.save_gifs:
                # image = self.envs.render('rgb_array')[0][0]
                # all_frames.append(image)
                self.envs.render('rgb_array')
                img = np.array(display.waitgrab())                
                all_frames.append(img)
                time.sleep(0.02)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                vels = []

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    # if not self.use_centralized_V:
                    #     share_obs = np.array(list(obs[:, agent_id]))
                    # self.trainer[agent_id].prep_rollout()
                    # action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                    #                                                     rnn_states[:, agent_id],
                    #                                                     masks[:, agent_id],
                    #                                                     deterministic=True)

                    # action = action.detach().cpu().numpy()
                    # # rearrange action
                    # if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    #     for i in range(self.envs.action_space[agent_id].shape):
                    #         uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    #         if i == 0:
                    #             action_env = uc_action_env
                    #         else:
                    #             action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    # elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    #     action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    # else:
                    #     raise NotImplementedError
                    adv_x = 4
                    adv_y = 5
                    adv_strategy = 'escape_nearest'
                    init_pos = [[-0.05, 0], [0.05, 0], [0, -0.05], [0, 0.05]]
                    guess_center = (0.5, 0.5)
                    action_env = np.zeros(shape=(1, 5))                    
                    if agent_id == 0:
                        vel = np.sqrt(np.sum(np.square([obs[0][agent_id][0], obs[0][agent_id][1]]))) / 0.05 * 1000
                        vels.append(vel)
                        if adv_strategy == 'random':
                            action_env[0][np.random.randint(5)]=1
                        elif adv_strategy == 'escape_nearest':
                            dists = []
                            for i in range(1, self.num_agents):
                                if obs[0][agent_id][2+2*i]==0 and obs[0][agent_id][3+2*i]==0:
                                    dist = 100 # mask
                                else:
                                    dist = np.sqrt(np.sum(np.square([obs[0][agent_id][2+2*i], obs[0][agent_id][3+2*i]])))
                                dists.append(dist)
                            nearest_index = np.argmin(np.array(dists)) + 1
                            if dists[nearest_index-1] == 100: # all good agents' position are masked
                                action_env[0][np.random.randint(5)]=1
                            else:
                                if abs(obs[0][agent_id][2+2*nearest_index]) > abs(obs[0][agent_id][3+2*nearest_index]):
                                    i = 1 if obs[0][agent_id][2+2*nearest_index] < 0 else 2 
                                    action_env[0][i]=1
                                else:
                                    i = 3 if obs[0][agent_id][3+2*nearest_index] < 0 else 4
                                    action_env[0][i]=1 

                        elif adv_strategy == 'escape_group':
                            dists = []
                            for i in range(1, self.num_agents):
                                if obs[0][agent_id][2+2*i]==0 and obs[0][agent_id][3+2*i]==0:
                                    dist = 100 # mask
                                else:
                                    dist = np.sqrt(np.sum(np.square([obs[0][agent_id][2+2*i], obs[0][agent_id][3+2*i]])))
                                dists.append(dist)
                            if min(dists) == 100:
                                action_env[0][np.random.randint(5)]=1
                            else:
                                dists_reciprocal = [1/x for x in dists]
                                dists_reciprocal_sum = np.sum(dists_reciprocal)
                                dists_reciprocal_norm = [x/dists_reciprocal_sum for x in dists_reciprocal]
                                for index in range(1, self.num_agents):
                                    if abs(obs[0][agent_id][2+2*index]) > abs(obs[0][agent_id][3+2*index]):
                                        i = 1 if obs[0][agent_id][2+2*index] < 0 else 2 
                                        action_env[0][i]+=1*dists_reciprocal_norm[index-1]
                                    else:
                                        i = 3 if obs[0][agent_id][3+2*index] < 0 else 4
                                        action_env[0][i]+=1*dists_reciprocal_norm[index-1]
                        else:
                            raise NotImplementedError("Unknown Strategy!")
                    else:
                        # if infos[0][agent_id]['detect_adversary'] == True:
                        #     action_env[0][0]=1
                        # else:
                        vel = np.sqrt(np.sum(np.square([obs[0][agent_id][0], obs[0][agent_id][1]]))) / 0.05 * 1000
                        vels.append(vel)
                        if obs[0][agent_id][adv_x] == 0 and obs[0][agent_id][adv_y] == 0:
                            obs[0][agent_id][adv_x] = guess_center[0] + init_pos[agent_id-1][0] - obs[0][agent_id][2]
                            obs[0][agent_id][adv_y] = guess_center[1] + init_pos[agent_id-1][1] - obs[0][agent_id][3]
                        if abs(obs[0][agent_id][adv_x]) > abs(obs[0][agent_id][adv_y]):
                            i = 1 if obs[0][agent_id][adv_x] > 0 else 2 
                            action_env[0][i]=1
                        else:
                            i = 3 if obs[0][agent_id][adv_y] > 0 else 4 
                            action_env[0][i]=1
                    temp_actions_env.append(action_env)
                    # rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)
                
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                agent_info = "agent{} :   velocity : {}m/s    detect the adversary : {}"
                adversary_info = "velocity : {}m/s    detect the adversary : {}"
                environment_info = "time : {} min {} s  success : {}    failure : {}    win rate : {}"
                if self.all_args.save_gifs:
                    # image = self.envs.render('rgb_array')[0][0]
                    # all_frames.append(image)
                    self.envs.render('rgb_array')
                    # all_frames.append(np.array(display.waitgrab()))
                    img = np.array(display.waitgrab())                
                    img_pil = Image.fromarray(img, mode='RGB')
                    ttf = ImageFont.load_default()
                    img_draw = ImageDraw.Draw(img_pil)
                    for i in range(self.num_agents):                        
                        img_draw.text((20, 20+15*i), agent_info.format(i, vels[i], infos[0][i]["detect_adversary"]), font=ttf, fill=(0, 0, 0))
                    img_draw.text((20, 95), environment_info.format(int(step*4.5/60), step*4.5%60, 0, 0, 0), font=ttf, fill=(0, 0, 0))
                    img_text = np.asarray(img_pil)
                    all_frames.append(img_text)
                    time.sleep(0.02)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)