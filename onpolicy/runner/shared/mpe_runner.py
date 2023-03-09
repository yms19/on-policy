import time
import numpy as np
import torch
from pyvirtualdisplay.smartdisplay import SmartDisplay
from PIL import Image, ImageDraw, ImageFont
import wandb
import imageio
import pdb

from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

def get_adv_action(num_agents, adv_strategy, obs, init_direction):
    action_env = np.zeros(7)
    if adv_strategy == 'random':
        action_env[np.random.randint(5)]=1
    elif adv_strategy == 'escape_nearest':
        dists = []
        for i in range(num_agents):
            if obs[4+2*i]==0 and obs[5+2*i]==0:
                dist = 100 # mask
            else:
                dist = np.sqrt(np.sum(np.square([obs[4+2*i], obs[5+2*i]])))
            dists.append(dist)
        nearest_index = np.argmin(np.array(dists))
        if dists[nearest_index] == 100: # all good agents' position are masked
            action_env[init_direction]=1
        else:
            if abs(obs[4+2*nearest_index]) > abs(obs[5+2*nearest_index]):
                i = 1 if obs[4+2*nearest_index] < 0 else 2 
                action_env[i]=1
            else:
                i = 3 if obs[5+2*nearest_index] < 0 else 4
                action_env[i]=1 

    elif adv_strategy == 'escape_group':
        dists = []
        for i in range(num_agents):
            if obs[4+2*i]==0 and obs[5+2*i]==0:
                dist = 100 # mask
            else:
                dist = np.sqrt(np.sum(np.square([obs[4+2*i], obs[5+2*i]])))
            dists.append(dist)
        if min(dists) == 100:
            action_env[init_direction]=1
        else:
            dists_reciprocal = [1/x for x in dists]
            dists_reciprocal_sum = np.sqrt(np.sum(np.square(dists_reciprocal)))
            dists_reciprocal_norm = [x/dists_reciprocal_sum for x in dists_reciprocal]
            for index in range(num_agents):
                if abs(obs[4+2*index]) > abs(obs[5+2*index]):
                    i = 1 if obs[4+2*index] < 0 else 2 
                    action_env[i]+=1*dists_reciprocal_norm[index]
                else:
                    i = 3 if obs[5+2*index] < 0 else 4
                    action_env[i]+=1*dists_reciprocal_norm[index]
    elif adv_strategy == "stop":
        action_env[0] = 1
    else:
        raise NotImplementedError("Unknown Strategy!")

    return action_env

# old version
def get_good_action(num_agents, obs, agent_id, step, avail_action):
    adv_x = 4
    adv_y = 5
    init_pos = [[-0.05, 0], [0.05, 0], [0, -0.05], [0, 0.05]]
    # target_pos = [[0.5, 0.5], [0.67, 0.67], [0.7, 0.4], [1, 0.8]]
    target_pos = [[obs[18], obs[19]]] * 4
    action_env = np.zeros(shape=(1, 7))
    detect_threshold = 0.02
    if obs[adv_x] == 0 and obs[adv_y] == 0:
        obs[adv_x] = target_pos[agent_id][0] + init_pos[agent_id][0] - obs[2]
        obs[adv_y] = target_pos[agent_id][1] + init_pos[agent_id][1] - obs[3]
    if np.sqrt(np.sum(np.square([obs[adv_x], obs[adv_y]]))) < detect_threshold:
        action_env[0][0]=1
    elif abs(obs[adv_x]) > abs(obs[adv_y]):
        i = 1 if obs[adv_x] > 0 else 2 
        action_env[0][i]=1
    else:
        i = 3 if obs[adv_y] > 0 else 4 
        action_env[0][i]=1
    
    return action_env

# new version
def get_good_action_with_detect(num_agents, obs, agent_id, step, avail_action):
    adv_x = 4
    adv_y = 5
    detect_threshold = 0.01
    init_pos = [[-0.05, 0], [0.05, 0], [0, -0.05], [0, 0.05]]
    adv_center = [obs[18], obs[19]]
    target_pos1 = [adv_center, [adv_center[0]+0.17, adv_center[1]+0.17], [adv_center[0]+0.5, adv_center[1]-0.1], [adv_center[0]+0.5, adv_center[1]+0.3]]
    target_pos2 = [[adv_center[0]-0.2, adv_center[1]], [adv_center[0]+0.17, adv_center[1]+0.57], [adv_center[0]+0.7, adv_center[1]-0.1], [adv_center[0]+0.7, adv_center[1]+0.3]]
    # target_pos = [[0.5, 0.5], [0.67, 0.67], [0.7, 0.4], [1, 0.8]]
    # target_pos = [[obs[18], obs[19]]] * 4
    action_env = np.zeros(shape=(1, 7))
    # print(obs)
    # print(target_pos)
    # exit()        
    # if step in range(80+agent_id*10, 120+agent_id*10):
    #     action_env[0][5]=1
    # elif step == 120+agent_id*10:
    #     action_env[0][6]=1
    # else:
    if obs[adv_x] == 0 and obs[adv_y] == 0 :
        if step < 173:
            obs[adv_x] = target_pos1[agent_id][0] + init_pos[agent_id][0] - obs[2]
            obs[adv_y] = target_pos1[agent_id][1] + init_pos[agent_id][1] - obs[3]
        else:
            obs[adv_x] = target_pos2[agent_id][0] + init_pos[agent_id][0] - obs[2]
            obs[adv_y] = target_pos2[agent_id][1] + init_pos[agent_id][1] - obs[3]

    # if np.sqrt(np.sum(np.square([obs[adv_x], obs[adv_y]]))) < detect_threshold and not first_achieve:
    if agent_id == 0 and step in range(88, 129):
        if step in range(88, 128):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1
    elif agent_id == 0 and step in range(185, 226):
        if step in range(185, 225):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1
    elif agent_id == 1 and step in range(106, 148):
        if step in range(106, 147):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1            
    elif agent_id == 1 and step in range(197, 238):
        if step in range(197, 237):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1  
    elif agent_id == 2 and step in range(121, 162):
        if step in range(121, 161):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1
    elif agent_id == 2 and step in range(188, 229):
        if step in range(188, 228):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1
    elif agent_id == 3 and step in range(131, 172):
        if step in range(131, 171):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1
    elif agent_id == 3 and step in range(188, 229):
        if step in range(188, 228):
            action_env[0][5] = 1
        else:
            action_env[0][6] = 1
    else:
        if abs(obs[adv_x]) > abs(obs[adv_y]) and avail_action[1]:
            i = 1 if obs[adv_x] > 0 else 2 
            action_env[0][i]=1
        elif avail_action[3]:
            i = 3 if obs[adv_y] > 0 else 4
            action_env[0][i]=1 
    
    return action_env

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.adv_obs = np.zeros((self.all_args.n_rollout_threads, 1, *self.envs.observation_space[0].shape))
        self.script_length = self.all_args.script_length
        self.num_adversaries = self.all_args.num_adversaries
        self.num_good_agents = self.all_args.num_good_agents
        self.num_agents = self.num_adversaries + self.num_good_agents
        self.role = ["adv", "good"]
        self.num_agents_role = {'adv': self.num_adversaries, 'good': self.num_good_agents}
        self.num_agents_range = {'adv':[0,self.num_adversaries-1],'good':[self.num_adversaries,self.num_adversaries+self.num_good_agents-1]}

        # dir
        self.model_dir = {}
        self.model_dir['adv'] = self.all_args.model_dir_role1
        self.model_dir['good'] = self.all_args.model_dir_role2

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = {'adv':Policy(self.all_args,
                    self.envs.observation_space[0],
                    share_observation_space,
                    self.envs.action_space[0],
                    device = self.device),
                'good':Policy(self.all_args,
                    self.envs.observation_space[0],
                    share_observation_space,
                    self.envs.action_space[self.num_adversaries],
                    device = self.device)}
        
        for role_id in self.role:
            if self.model_dir[role_id] is not None:
                self.restore(role_id)
        
        # algorithm
        self.trainer = {'adv':TrainAlgo(self.all_args, self.policy['adv'], device = self.device),
                        'good':TrainAlgo(self.all_args, self.policy['good'], device = self.device)}        
        
        # buffer
        self.buffer = {'adv': SharedReplayBuffer(self.all_args,
                                    self.num_adversaries,
                                    self.envs.observation_space[0],
                                    share_observation_space,
                                    self.envs.action_space[0]),
                        'good': SharedReplayBuffer(self.all_args,
                                    self.num_good_agents,
                                    self.envs.observation_space[0],
                                    share_observation_space,
                                    self.envs.action_space[self.num_adversaries])}        

    def restore(self, role):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir[role]) + '/actor_%s.pt' % role)
        self.policy[role].actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir[role]) + '/critic_%s.pt' % role)
            self.policy[role].critic.load_state_dict(policy_critic_state_dict)

    def save(self, role):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer[role].policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_%s.pt" % role)
        policy_critic = self.trainer[role].policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_%s.pt" % role)

    def compute(self, role=None):
        """Calculate returns for the collected data."""
        self.trainer[role].prep_rollout()
        next_values = self.trainer[role].policy.get_values(np.concatenate(self.buffer[role].share_obs[-1]),
                                                np.concatenate(self.buffer[role].rnn_states_critic[-1]),
                                                np.concatenate(self.buffer[role].masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer[role].compute_returns(next_values, self.trainer[role].value_normalizer)
    
    def train(self, role=None):
        """Train policies with data in buffer. """
        self.trainer[role].prep_training()
        train_infos = self.trainer[role].train(self.buffer[role], role)      
        self.buffer[role].after_update()
        return train_infos        

    def run(self):
        self.warmup()
        # obs = self.buffer.obs[0]
        # available_actions = self.buffer.available_actions[0] 

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        win_count = 0
        fail_count = 0
        adv_strategy = 'escape_nearest'

        for episode in range(episodes):
            init_direction = np.random.randint(4, size=(self.all_args.n_rollout_threads)) + 1
            win = np.zeros((self.n_rollout_threads))
            win_step = np.zeros((self.n_rollout_threads))
            if self.use_linear_lr_decay:
                for role_id in self.role:
                    self.trainer[role_id].policy.lr_decay(episode, episodes)          
                
            for step in range(self.episode_length+self.script_length):
                values_roles = {}
                actions_roles = {}
                action_log_probs_roles = {}
                rnn_states_roles = {}
                rnn_states_critic_roles = {}
                actions_env_roles = []
                # Sample actions
                if step < self.script_length:
                    actions_env = np.zeros([self.all_args.n_rollout_threads, self.num_agents, 7])
                    for thread_index in range(self.all_args.n_rollout_threads):
                        for agent_index in range(self.num_agents):
                            actions_env[thread_index][agent_index] = get_good_action(self.num_good_agents, obs[thread_index][agent_index-self.num_agents], agent_index, step, available_actions[thread_index][agent_index-self.num_agents])
                            # print("step{} obs of agent{}: ({}, {})".format(step, agent_index, obs[thread_index][agent_index-self.num_agents][2], obs[thread_index][agent_index-self.num_agents][3]))
                    # print("step%d action"%step, np.argmax(actions_env[0][0]), np.argmax(actions_env[0][1]), np.argmax(actions_env[0][2]), np.argmax(actions_env[0][3]))
                else:
                    for role_id in self.role:
                        values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step-self.script_length, role=role_id)
                        values_roles[role_id] = values
                        actions_roles[role_id] = actions
                        action_log_probs_roles[role_id] = action_log_probs
                        rnn_states_roles[role_id] = rnn_states
                        rnn_states_critic_roles[role_id] = rnn_states_critic
                        actions_env_roles.append(actions_env)

                actions_env_all = np.concatenate(actions_env_roles,axis=1)

                # pdb.set_trace()
                    
                # Obser reward and next obs
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env_all)
                self.adv_obs = obs[:, 0, :].copy()

                if step == self.script_length - 1:
                    if self.use_centralized_V:
                        share_obs = obs.reshape(self.n_rollout_threads, -1)
                        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
                    else:
                        share_obs = obs

                    self.buffer.share_obs[0] = share_obs.copy()
                    self.buffer.obs[0] = obs[:, 1:, :].copy()
                    self.buffer.available_actions[0] = available_actions[:, 1:, :].copy()

                for thread_index in range(self.all_args.n_rollout_threads):
                    for agent_index in range(self.num_adversaries,self.num_agents):
                        if infos[thread_index][agent_index]['detect_adversary']:
                            win[thread_index] = 1
                            if win_step[thread_index]==0:
                                win_step[thread_index] = step+1
                # print("step%d action"%step, actions_env_all[0][0], actions_env_all[0][1], actions_env_all[0][2], actions_env_all[0][3], actions_env_all[0][4])
                # print("step%d reward"%step, rewards[0][0], rewards[0][1], rewards[0][2], rewards[0][3], rewards[0][4])
                # print("step%d avail"%step,  available_actions[0][1], available_actions[0][2], available_actions[0][3], available_actions[0][4])

                # for i in range(self.num_agents):
                #     print("eval average episode rewards of agent%i: " % i + str(rewards[:, :,]))

                # print("step%d obs"%step, obs[0][1],obs[0][2],obs[0][3],obs[0][3])
                if step >= self.script_length:
                    infos_role = {}
                    for role_id in self.role:
                        role_range = self.num_agents_range[role_id]
                        infos_role_one = []
                        for thread in range(self.n_rollout_threads):
                            infos_role_one.append(infos[thread][role_range[0]:role_range[1]+1])
                        infos_role[role_id] = infos_role_one
                        # use all agents obs to get critic share obs
                        data = obs, rewards[:,role_range[0]:role_range[1]+1], dones[:,role_range[0]:role_range[1]+1], infos_role[role_id], available_actions[:,role_range[0]:role_range[1]+1], \
                                values_roles[role_id], actions_roles[role_id], action_log_probs_roles[role_id], rnn_states_roles[role_id], rnn_states_critic_roles[role_id]

                        # insert data into buffer
                        self.insert(data, role=role_id)
                # print("finish episode {} step {}".format(episode,step))              
            # for i in range(self.num_agents):
            #     average_episode_rewards = np.sum(self.buffer.rewards[:, :, i])
            #     print("eval average episode rewards of agent%i: " % i + str(average_episode_rewards))

            # compute return and update network
            win_count += np.sum(win)
            fail_count += self.all_args.n_rollout_threads - np.sum(win)
            # print("episode{} average episode rewards is {}".format(episode, np.mean(self.buffer.rewards) * self.episode_length))
            train_infos = {}
            for role_id in self.role:
                self.compute(role=role_id)
                train_infos[role_id] = self.train(role=role_id)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                # print("\nModel save to {}".format(self.save_dir))
                for role_id in self.role:
                    self.save(role_id)

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
                        agent_k = 'agent%i/individual_rewards' % (agent_id)
                        env_infos[agent_k] = idv_rews

                for role_id in self.role:
                    train_infos[role_id]["average_episode_rewards"] = np.mean(self.buffer[role_id].rewards) * self.episode_length
                    print("average_episode_rewards of " + role_id + " is {}".format(train_infos[role_id]["average_episode_rewards"]))
                train_infos["good"]["win_rate"] = win_count / (win_count + fail_count)
                train_infos["good"]["average_detect_step"] = np.sum(win_step)/win_count
                print("win rate is {:.2f}%".format(train_infos["win_rate"]*100))
                for role_id in self.role:
                    self.log_train(train_infos[role_id], total_num_steps)
                self.log_env(env_infos, total_num_steps)
                win_count = 0
                fail_count = 0

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, available_actions = self.envs.reset()
        obs_adv = obs[:, 0:self.num_adversaries,:]
        obs_good = obs[:, self.num_adversaries:, :]
        avail_adv = available_actions[:, 0:self.num_adversaries,:]
        avail_good = available_actions[:, self.num_adversaries:, :]      

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        share_obs_adv = share_obs[:, 0:self.num_adversaries,:]
        share_obs_good = share_obs[:, self.num_adversaries:, :] 

        self.buffer["adv"].share_obs[0] = share_obs_adv.copy()
        self.buffer["adv"].obs[0] = obs_adv.copy()
        self.buffer["adv"].available_actions[0] = avail_adv.copy()
        self.buffer["good"].share_obs[0] = share_obs_good.copy()
        self.buffer["good"].obs[0] = obs_good.copy()
        self.buffer["good"].available_actions[0] = avail_good.copy()

    @torch.no_grad()
    def collect(self, step, role=None):
        self.trainer[role].prep_rollout()
        role_range = self.num_agents_range[role]
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer[role].policy.get_actions(np.concatenate(self.buffer[role].share_obs[step]),
                            np.concatenate(self.buffer[role].obs[step]),
                            np.concatenate(self.buffer[role].rnn_states[step]),
                            np.concatenate(self.buffer[role].rnn_states_critic[step]),
                            np.concatenate(self.buffer[role].masks[step]),
                            np.concatenate(self.buffer[role].available_actions[step]),
                            )
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
            actions_env = np.squeeze(np.eye(self.envs.action_space[role_range[0]].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data, role=None):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer[role].rnn_states_critic.shape[3:]), dtype=np.float32)
        if role=='adv':
            masks = np.ones((self.n_rollout_threads, self.num_adversaries, 1), dtype=np.float32)
        elif role=='good':
            masks = np.ones((self.n_rollout_threads, self.num_good_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        role_range = self.num_agents_range[role]

        self.buffer[role].insert(share_obs[:,role_range[0]:role_range[1]+1], obs[:,role_range[0]:role_range[1]+1], rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions)

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
        step_count = 0
        accumulate_reward = np.zeros((self.all_args.num_good_agents))
        for episode in range(self.all_args.render_episodes):
            obs, avail_actions = envs.reset()            
            init_direction = np.random.randint(4, size=(self.all_args.n_rollout_threads)) + 1
            win = np.zeros((self.n_rollout_threads))
            win_step = np.zeros((self.n_rollout_threads))
            print("init_pos: ({}, {})".format(obs[0][0][2], obs[0][0][3]))
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
            rnn_states_roles = {}
            masks_roles = {}
            for role_id in self.role:
                role_range = self.num_agents_role[role_id]
                role_num = self.num_agents_role[role_id]
                rnn_states_roles[role_id] = np.zeros((self.n_rollout_threads, role_num, self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks_roles[role_id] = np.ones((self.n_rollout_threads, role_num, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length+self.script_length):
                calc_start = time.time()
                adv_strategy = 'escape_group'
                mode = "scripts"
                vels = {}
                actions_env_roles = []

                for role_id in self.role:
                    role_range = self.num_agents_range[role_id]
                    vels[role_id] = []
                    if self.model_dir[role_id] is not None:
                        if step < self.script_length:
                            actions_env = np.zeros([self.all_args.n_rollout_threads, self.num_agents, 7])
                            for thread_index in range(self.all_args.n_rollout_threads):
                                for agent_index in range(self.num_agents):
                                    actions_env[thread_index][agent_index] = get_good_action(self.num_agents, obs[thread_index][agent_index], agent_index, step, avail_actions[thread_index][agent_index])
                                    # print("step{} obs of agent{}: ({}, {})".format(step, agent_index, obs[thread_index][agent_index][2], obs[thread_index][agent_index][3]))
                            # print("step%d action"%step, np.argmax(actions_env[0][0]), np.argmax(actions_env[0][1]), np.argmax(actions_env[0][2]), np.argmax(actions_env[0][3]))
                        else:
                            self.trainer[role_id].prep_rollout()
                            action, rnn_states = self.trainer[role_id].policy.act(np.concatenate(obs[:,role_range[0]:role_range[1]+1 ,:]),
                                                                np.concatenate(rnn_states_roles[role_id]),
                                                                np.concatenate(masks_roles[role_id]),
                                                                np.concatenate(avail_actions[:,role_range[0]:role_range[1]+1 ,:]),
                                                                deterministic=True)
                            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
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
                            rnn_states_roles[role_id] = rnn_states
                            
                            for thread_index in range(self.n_rollout_threads):                            
                                for agent_index in range(role_range[0], role_range[1]+1):
                                    vel = np.sqrt(np.sum(np.square([obs[thread_index][agent_index][0], obs[thread_index][agent_index][1]]))) / 0.05 * 1000
                                    vels[role_id].append(vel)
                        # pdb.set_trace()


                    else:
                        role_num = self.num_agents_role[role_id]
                        actions_env = np.zeros((self.n_rollout_threads, role_num, 7))
                        for thread_index in range(self.n_rollout_threads):
                            if role_id == "adv":
                                for agent_index in range(role_range[0], role_range[1]+1):
                                    action = get_adv_action(self.num_agents, adv_strategy, obs[thread_index][agent_index],  init_direction[thread_index])
                                    actions_env[thread_index][agent_index] = action
                                    vel = np.sqrt(np.sum(np.square([obs[thread_index][agent_index][0], obs[thread_index][agent_index][1]]))) / 0.05 * 1000
                                    vels[role_id].append(vel)
                            elif role_id == "good":
                                for agent_index in range(role_range[0], role_range[1]+1):
                                    action = get_good_action_with_detect(self.num_agents, obs[thread_index][agent_index], agent_index, step, avail_actions[thread_index][agent_index])
                                    actions_env[thread_index][agent_index-self.num_adversaries] = action
                                    vel = np.sqrt(np.sum(np.square([obs[thread_index][agent_index][0], obs[thread_index][agent_index][1]]))) / 0.05 * 1000
                                    vels[role_id].append(vel)
                            else:
                                raise NotImplementedError
                    actions_env_roles.append(actions_env)

                # Obser reward and next obs
                actions_env_all = np.concatenate(actions_env_roles,axis=1)
                obs, rewards, dones, infos, avail_actions = envs.step(actions_env_all)
                # print("step%d reward"%step, rewards[0][0], rewards[0][1], rewards[0][2], rewards[0][3], rewards[0][4])
                # print("step%d avail"%step,  avail_actions[0][1], avail_actions[0][2], avail_actions[0][3], avail_actions[0][4])
                # print("step%d"%step)
                # print(obs[0][1],obs[0][2],obs[0][3],obs[0][3])

                for thread_index in range(self.n_rollout_threads):
                    for agent_index in range(self.num_good_agents):
                        if infos[thread_index][agent_index+self.num_adversaries]['detect_adversary'] and win_step[thread_index]==0:
                                win[thread_index] = 1
                                win_step[thread_index] = step+1

                episode_rewards.append(rewards[:, self.num_adversaries:, :])
                
                # print("step%d reward"%step, rewards[0][1], rewards[0][2], rewards[0][3], rewards[0][4])
                for role_id in self.role:
                    role_range = self.num_agents_range[role_id]
                    role_num = self.num_agents_role[role_id]

                    rnn_states_roles[role_id][dones[:, role_range[0]:role_range[1]+1] == True] = np.zeros(((dones[:, role_range[0]:role_range[1]+1] == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                    masks = np.ones((self.n_rollout_threads, role_num, 1), dtype=np.float32)
                    masks[dones[:, role_range[0]:role_range[1]+1] == True] = np.zeros(((dones[:, role_range[0]:role_range[1]+1] == True).sum(), 1), dtype=np.float32)

                    masks_roles[role_id] = masks

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
                            img_draw.text((20, 20+15*i), adversary_info.format(i, vels['adv'][i], infos[0][i]["detect_adversary"], mode), font=ttf, fill=(0, 0, 0))
                        else:
                            # print(i, vels, infos)                      
                            img_draw.text((20, 20+15*i), agent_info.format(i, vels['good'][i-1], infos[0][i]["detect_times"], infos[0][i]["detect_adversary"]), font=ttf, fill=(0, 0, 0))
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
            win_count += np.sum(win)
            fail_count += self.n_rollout_threads - np.sum(win)

            step_count += np.sum(win_step)

            episode_rewards = np.array(episode_rewards)
            print("result of episode %i:" % episode)
            for agent_id in range(self.num_good_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % (agent_id+1) + str(average_episode_rewards))
                accumulate_reward[agent_id] += average_episode_rewards

        if self.all_args.save_gifs:
            print(self.gif_dir)
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
        
        print("win: {}\nfail: {}\nwin rate: {} %".format(win_count, fail_count, 100*win_count/(win_count+fail_count)))
        print("The average number of probe steps consumed is {}".format(step_count/win_count))
        for agent_id in range(self.num_good_agents):
            print("accumulative average episode rewards of agent%i: " % (agent_id+1) + str(accumulate_reward[agent_id]/self.all_args.render_episodes))
        
