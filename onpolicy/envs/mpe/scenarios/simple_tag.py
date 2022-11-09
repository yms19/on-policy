import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario

def hit(bubble, world):
    bubble.action.u = np.zeros(world.dim_p)

    def ready_to_hit(agent1, agent2):
        hit_range = 0.1
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < hit_range else False         

    if ready_to_hit(bubble.holder, world.agents[0]) or bubble.launched:
        bubble.max_speed = 0.001
        bubble.launched = True
        dis_x = (bubble.state.p_pos[0] - world.agents[0].state.p_pos[0])
        dis_y = (bubble.state.p_pos[1] - world.agents[0].state.p_pos[1])

        if abs(dis_x) > abs(dis_y) :
            if dis_x < 0:
                bubble.action.u[0] += 1.0
            elif dis_x > 0:
                bubble.action.u[0] -= 1.0
        else:
            if dis_y < 0:
                bubble.action.u[1] += 1.0
            elif dis_y > 0:
                bubble.action.u[1] -= 1.0

        sensitivity = 5.0
        if bubble.accel is not None:
            sensitivity = bubble.accel
            bubble.action.u *= sensitivity 
    # doesn't detect the adversary: follow the holder
    else:
        bubble.action.u[0] = bubble.holder.action.u[0]
        bubble.action.u[1] = bubble.holder.action.u[1]

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = args.num_good_agents#1
        num_adversaries = args.num_adversaries#3
        num_bubbles = args.num_bubbles
        # num_weapon = args.weapon
        num_agents = num_adversaries + num_good_agents + num_bubbles
        # num_agents = num_adversaries + num_good_agents + num_weapon
        num_landmarks = args.num_landmarks#2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.world_length = args.episode_length
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.dummy = True if i >= num_adversaries + num_good_agents else False
            agent.size = 0.025 if agent.adversary else 0.015 if not agent.dummy else 0.005
            agent.accel = 0.005 if agent.adversary else 0.012 # if not agent.dummy else 5.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 0.0003 if agent.adversary else 0.004 # if not agent.dummy else 1.5
            agent.d_range = 2 * args.d_range if agent.adversary else args.d_range if not agent.dummy else None
            agent.action_callback = hit if agent.dummy else None
            agent.holder = world.agents[i-num_good_agents] if agent.dummy else None
            agent.launched = False if agent.dummy else None
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.005
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        # random properties for landmarks
        # set random initial states
        init_pos = [[-1.05, 0.5], [-0.95, 0.5], [-1, 0.45], [-1, 0.55]]
        init_range = 0.4
        init_bias = [0.5, 0.5]
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                x = np.random.uniform(-1, +1) * init_range
                y = np.random.uniform(-1, +1) * init_range
                while x**2 + y**2 > init_range**2:
                    x = np.random.uniform(-1, +1) * init_range
                    y = np.random.uniform(-1, +1) * init_range
                x += init_bias[0]
                y += init_bias[1]
                agent.state.p_pos = np.array([x, y], dtype=float)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif agent.dummy:
                agent.state.p_pos = np.array(init_pos[i-5], dtype=float)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

            else:
                agent.state.p_pos = np.array(init_pos[i-1], dtype=float)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                # landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
                # landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.state.p_pos = np.array([0.5, 0.5])
                landmark.state.p_vel = np.zeros(world.dim_p)
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     if not landmark.boundary:
        #         landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
        #         landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def is_detected(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < agent1.d_range else False

    def ready_to_hit(self, agent1, agent2):
        hit_range = 0.2
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return True if dist < hit_range else False 

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False #different from openai
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False #different from openai
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        if agent.adversary:
            for other in world.agents:
                if other is agent or other.dummy: continue
                mask = 1 if not self.is_detected(agent, other) else 0
                comm.append(other.state.c)
                other_pos.append((other.state.p_pos - agent.state.p_pos) * (1 - mask))
                if not other.adversary:
                    other_vel.append((other.state.p_vel) * (1 - mask))
            other_vel = other_vel[:-1]
        else:
            for other in world.agents:
                if other is agent or other.dummy: continue
                mask = 1 if other.adversary and not self.is_detected(agent, other) else 0
                comm.append(other.state.c)
                other_pos.append((other.state.p_pos - agent.state.p_pos) * (1 - mask))
                if not other.adversary:
                    other_vel.append(other.state.p_vel)

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel)
    
    def info(self, agent, world):
        info = {'detect_adversary' : False}
        if not agent.adversary:
            info['detect_adversary'] = self.is_detected(agent, world.agents[0])
        else:
            info['detect_adversary'] = self.is_detected(agent, world.agents[1]) or self.is_detected(agent, world.agents[2]) \
                                        or self.is_detected(agent, world.agents[3]) or self.is_detected(agent, world.agents[4])
        
        return info
