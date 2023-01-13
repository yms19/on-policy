import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from pyvirtualdisplay.smartdisplay import SmartDisplay
from PIL import Image, ImageDraw, ImageFont
import imageio
import pdb

def _t2n(x):
    return x.detach().cpu().numpy()

def get_adv_action(num_agents, adv_strategy, obs, init_direction):
    action_env = np.zeros(7)
    if adv_strategy == 'random':
        action_env[np.random.randint(5)]=1
    elif adv_strategy == 'escape_nearest':
        dists = []
        for i in range(1, num_agents):
            if obs[2+2*i]==0 and obs[3+2*i]==0:
                dist = 100 # mask
            else:
                dist = np.sqrt(np.sum(np.square([obs[2+2*i], obs[3+2*i]])))
            dists.append(dist)
        nearest_index = np.argmin(np.array(dists)) + 1
        if dists[nearest_index-1] == 100: # all good agents' position are masked
            action_env[init_direction]=1
        else:
            if abs(obs[2+2*nearest_index]) > abs(obs[3+2*nearest_index]):
                i = 1 if obs[2+2*nearest_index] < 0 else 2 
                action_env[i]=1
            else:
                i = 3 if obs[3+2*nearest_index] < 0 else 4
                action_env[i]=1 

    elif adv_strategy == 'escape_group':
        dists = []
        for i in range(1, num_agents):
            if obs[2+2*i]==0 and obs[3+2*i]==0:
                dist = 100 # mask
            else:
                dist = np.sqrt(np.sum(np.square([obs[2+2*i], obs[3+2*i]])))
            dists.append(dist)
        if min(dists) == 100:
            action_env[init_direction]=1
        else:
            dists_reciprocal = [1/x for x in dists]
            dists_reciprocal_sum = np.sum(dists_reciprocal)
            dists_reciprocal_norm = [x/dists_reciprocal_sum for x in dists_reciprocal]
            for index in range(1, num_agents):
                if abs(obs[2+2*index]) > abs(obs[3+2*index]):
                    i = 1 if obs[2+2*index] < 0 else 2 
                    action_env[i]+=1*dists_reciprocal_norm[index-1]
                else:
                    i = 3 if obs[3+2*index] < 0 else 4
                    action_env[i]+=1*dists_reciprocal_norm[index-1]
    else:
        raise NotImplementedError("Unknown Strategy!")

    return action_env

def get_good_action(num_agents, obs, agent_id, step):
    adv_x = 4
    adv_y = 5
    init_pos = [[-0.05, 0], [0.05, 0], [0, -0.05], [0, 0.05]]
    target_pos = [[0.5, 0.5], [0.67, 0.67], [0.7, 0.4], [1, 0.8]]
    action_env = np.zeros(shape=(1, num_agents+2))        
    if step in range(80+agent_id*10, 120+agent_id*10):
        action_env[0][5]=1
    elif step == 120+agent_id*10:
        action_env[0][6]=1
    else:
        if obs[adv_x] == 0 and obs[adv_y] == 0:
            obs[adv_x] = target_pos[agent_id-1][0] + init_pos[agent_id-1][0] - obs[2]
            obs[adv_y] = target_pos[agent_id-1][1] + init_pos[agent_id-1][1] - obs[3]
        if abs(obs[adv_x]) > abs(obs[adv_y]):
            i = 1 if obs[adv_x] > 0 else 2 
            action_env[0][i]=1
        else:
            i = 3 if obs[adv_y] > 0 else 4 
        action_env[0][i]=1
    
    return action_env

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.adv_obs = np.zeros((self.all_args.n_rollout_threads, 1, *self.envs.observation_space[0].shape))

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            init_direction = np.random.randint(4, size=(self.all_args.n_rollout_threads)) + 1
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                for thread_index in range(self.all_args.n_rollout_threads):
                    actions_env[thread_index][0] = get_adv_action(self.all_args.num_agents, "escape_nearest",
                                                                self.adv_obs[thread_index], init_direction[thread_index])
                # pdb.set_trace()
                    
                # Obser reward and next obs
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)

                # print("step%d"%step)
                # print(obs[0][1],obs[0][2],obs[0][3],obs[0][3])

                data = obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                # print("\nModel save to {}".format(self.save_dir))
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
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, available_actions = self.envs.reset()
        self.adv_obs = obs[:, 0, :].copy()
        obs[:, 0, :] = obs[:, 0, :] * 0

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs[:, 1:, :].reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        self.adv_obs = obs[:, 0, :].copy()
        obs[:, 0, :] = obs[:, 0, :] * 0

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs[:, 1:, :].reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                np.concatenate(eval_available_actions),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self, display):
        """Visualize the env."""
        envs = self.envs       
        all_frames = []
        win_count = 0
        fail_count = 0
        for episode in range(self.all_args.render_episodes):
            obs, avail_actions = envs.reset()
            self.adv_obs = obs[:, 0, :].copy()
            obs[:, 0, :] = obs[:, 0, :] * 0
            init_direction = np.random.randint(4, size=(self.all_args.n_rollout_threads)) + 1
            win = False
            print("init_pos: ({}, {})".format(self.adv_obs[0][2], self.adv_obs[0][3]))
            infos = [[{'individual_reward': 0, 'detect_times': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_times': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_times': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_times': 0, 'detect_adversary': False},
                    {'individual_reward': 0, 'detect_times': 0, 'detect_adversary': False}]]
            if self.all_args.save_gifs:
                # image = envs.render('rgb_array')[0][0]
                # all_frames.append(image)
                envs.render('rgb_array')
                all_frames.append(np.array(display.waitgrab()))
                time.sleep(0.02)
            # else:
            #     envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()
                adv_strategy = 'escape_nearest'
                mode = "scripts"
                vels = []

                if self.all_args.model_dir is not None:
                    self.trainer.prep_rollout()
                    action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                        np.concatenate(rnn_states),
                                                        np.concatenate(masks),
                                                        np.concatenate(avail_actions),
                                                        deterministic=True)
                    actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                    # print("step%d action"%step, actions[0][1], actions[0][2], actions[0][3], actions[0][4])
                    rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                    if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                        for i in range(envs.action_space[0].shape):
                            uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                            if i == 0:
                                actions_env = uc_actions_env
                            else:
                                actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                    elif envs.action_space[0].__class__.__name__ == 'Discrete':
                        actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                    else:
                        raise NotImplementedError

                    for thread_index in range(self.n_rollout_threads):
                        adv_action = get_adv_action(self.num_agents, adv_strategy, self.adv_obs[thread_index], init_direction[thread_index])
                        actions_env[thread_index][0] = adv_action
                        vel = np.sqrt(np.sum(np.square([obs[thread_index][0][0], obs[thread_index][0][1]]))) / 0.05 * 1000
                        vels.append(vel)
                        for agent_index in range(1, self.num_agents):
                            vel = np.sqrt(np.sum(np.square([obs[thread_index][agent_index][0], obs[thread_index][agent_index][1]]))) / 0.05 * 1000
                            vels.append(vel)
                    # pdb.set_trace()


                else:
                    actions_env = np.zeros(shape=(self.n_rollout_threads, self.num_agents, 7))
                    for thread_index in range(self.n_rollout_threads):
                        adv_action = get_adv_action(self.num_agents, adv_strategy, self.adv_obs[thread_index],  init_direction[thread_index])
                        actions_env[thread_index][0] = adv_action
                        vel = np.sqrt(np.sum(np.square([obs[thread_index][0][0], obs[thread_index][0][1]]))) / 0.05 * 1000
                        vels.append(vel)
                        for agent_index in range(1, self.num_agents):
                            action = get_good_action(self.num_agents, obs[thread_index][agent_index], agent_index, step)
                            actions_env[thread_index][agent_index] = action
                            vel = np.sqrt(np.sum(np.square([obs[thread_index][agent_index][0], obs[thread_index][agent_index][1]]))) / 0.05 * 1000
                            vels.append(vel)

                # actions_env = np.zeros(shape=(self.n_rollout_threads, self.num_agents, 5))

                # Obser reward and next obs
                obs, rewards, dones, infos, avail_actions = envs.step(actions_env)
                self.adv_obs = obs[:, 0, :].copy()
                obs[:, 0, :] = obs[:, 0, :] * 0
                episode_rewards.append(rewards)

                for i in range(1, self.num_agents):
                    if infos[0][i]["detect_adversary"]:
                        win = True
                
                # print("step%d reward"%step, rewards[0][1], rewards[0][2], rewards[0][3], rewards[0][4])

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                agent_info = "agent{} :   velocity : {:.2f}m/s    detect times : {}    detect the adversary : {}"
                adversary_info = "agent{} :   (adversary) velocity : {:.2f}m/s    detect the adversary : {}    mode : {}"
                environment_info = "time : {} min {} s  success : {}    failure : {}"

                if self.all_args.save_gifs:
                    # image = envs.render('rgb_array')[0][0]
                    # all_frames.append(image)
                    envs.render('rgb_array')
                    img = np.array(display.waitgrab())                
                    img_pil = Image.fromarray(img, mode='RGB')
                    ttf = ImageFont.load_default()
                    img_draw = ImageDraw.Draw(img_pil)
                    for i in range(self.num_agents):  
                        if i == 0:
                            img_draw.text((20, 20+15*i), adversary_info.format(i, vels[i], infos[0][i]["detect_adversary"], mode), font=ttf, fill=(0, 0, 0))
                        else:                      
                            img_draw.text((20, 20+15*i), agent_info.format(i, vels[i], infos[0][i]["detect_times"], infos[0][i]["detect_adversary"]), font=ttf, fill=(0, 0, 0))
                    img_draw.text((20, 95), environment_info.format(int(step*4.5/60), step*4.5%60, win_count, fail_count),
                            font=ttf, fill=(0, 0, 0))
                    img_text = np.asarray(img_pil)
                    all_frames.append(img_text)
                    time.sleep(0.02)

                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                # else:
                #     envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            if win:
                win_count += 1
            else:
                fail_count += 1

            episode_rewards = np.array(episode_rewards)
            print("\nresult of episode %i:" % episode)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            print(self.gif_dir)
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
        
        print("win: {}\nfail: {}\nwin rate: {} %".format(win_count, fail_count, 100*win_count/(win_count+fail_count)))
        