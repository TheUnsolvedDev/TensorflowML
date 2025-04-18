import numpy as np

from config import *

class ReplayBuffer:
    def __init__(self,buffer_size=BUFFER_SIZE,num_envs=NUM_ENVS):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.buffer = []
        self.index = 0
        
    def __len__(self):
        return len(self.buffer)
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.index] = (state, action, reward, next_state, done)
            self.index = (self.index + 1) % self.buffer_size
        
    def sample(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to sample from.")
        indices = np.random.choice(len(self.buffer), size=int(batch_size/self.num_envs), replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        if self.num_envs > 1:
            states = np.concatenate(states, axis=0)
            actions = np.concatenate(actions, axis=0)
            rewards = np.concatenate(rewards, axis=0)
            next_states = np.concatenate(next_states, axis=0)
            dones = np.concatenate(dones, axis=0)
        else:
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
        return states, actions, rewards, next_states, dones