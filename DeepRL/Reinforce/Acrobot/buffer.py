import numpy as np

from config import *

class TrajectoryBuffer:
    def __init__(self, num_envs=NUM_ENVS):
        self.num_envs = num_envs
        self.buffer = [[] for _ in range(num_envs)]

    def add(self, state, action, reward):
        if self.num_envs == 1:
            self.buffer[0].append((state, action, reward))
            return self.buffer
        for i in range(self.num_envs):
            self.buffer[i].append((state[i], action[i], reward[i]))
        return self.buffer

    def get_buffer(self, env_id):
        trajectory = self.buffer[env_id]
        states, actions, rewards = zip(*trajectory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        return states, actions, rewards

    def clear(self, env_id):
        self.buffer[env_id] = []
        
    def reset(self):
        for i in range(self.num_envs):
            self.clear(i)