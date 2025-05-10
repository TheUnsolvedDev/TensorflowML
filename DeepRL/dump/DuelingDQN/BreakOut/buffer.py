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
    
def multi_env_test():
    buffer = ReplayBuffer(buffer_size=BUFFER_SIZE,num_envs=NUM_ENVS)
    state = np.random.rand(NUM_ENVS,84,84,NUM_CHANNELS)
    action = np.random.randint(0, NUM_ACTIONS, size=(NUM_ENVS,))
    reward = np.random.rand(NUM_ENVS,)
    next_state = np.random.rand(NUM_ENVS,84,84,NUM_CHANNELS)
    done = np.random.randint(0, 2, size=(NUM_ENVS,))
    
    for _ in range(100):
        buffer.add(state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=BATCH_SIZE)
    
    assert states.shape == (BATCH_SIZE, 84, 84, NUM_CHANNELS)
    assert actions.shape == (BATCH_SIZE,)
    assert rewards.shape == (BATCH_SIZE,)
    assert next_states.shape == (BATCH_SIZE, 84, 84, NUM_CHANNELS)
    assert dones.shape == (BATCH_SIZE,)
    print("Multi-env test passed!")
    
def single_env_test():
    buffer = ReplayBuffer(buffer_size=BUFFER_SIZE,num_envs=1)
    state = np.random.rand(84,84,NUM_CHANNELS)
    action = np.random.randint(0, NUM_ACTIONS, size=(1,))[0]
    reward = np.random.rand(1,)[0]
    next_state = np.random.rand(84,84,NUM_CHANNELS)
    done = np.random.randint(0, 2, size=(1,))[0]
    
    for _ in range(100):
        buffer.add(state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=BATCH_SIZE)
    
    assert states.shape == (BATCH_SIZE, 84, 84, NUM_CHANNELS)
    assert actions.shape == (BATCH_SIZE,)
    assert rewards.shape == (BATCH_SIZE,)
    assert next_states.shape == (BATCH_SIZE, 84, 84, NUM_CHANNELS)
    assert dones.shape == (BATCH_SIZE,)
    print("Single-env test passed!")
    
if __name__ == "__main__":
    multi_env_test()
    single_env_test()