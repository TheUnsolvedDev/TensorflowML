import numpy as np

from config import *


class ReplayBuffer:
    def __init__(self, batch_size=BATCH_SIZE, max_size=MAX_CAPACITY):
        self.batch_size = batch_size
        self.max_size = max_size

        self.buffer = {
            'state': np.zeros((self.max_size, STATE_SHAPE[0], STATE_SHAPE[1], STACK_FRAMES), dtype=np.float32),
            'action': np.zeros((self.max_size, 1), dtype=np.int8),
            'reward': np.zeros((self.max_size, 1), dtype=np.float32),
            'next_state': np.zeros((self.max_size, STATE_SHAPE[0], STATE_SHAPE[1], STACK_FRAMES), dtype=np.float32),
            'done': np.zeros((self.max_size, 1), dtype=np.int8)
        }
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if self.position >= self.max_size:
            self.position = 0
        self.buffer['state'][self.position] = state
        self.buffer['action'][self.position] = action
        self.buffer['reward'][self.position] = reward
        self.buffer['next_state'][self.position] = next_state
        self.buffer['done'][self.position] = done
        self.position += 1

    def get(self):
        if self.position >= self.batch_size:
            indices = np.random.randint(0, self.position, self.batch_size)
        else:
            indices = np.random.randint(0, self.position, self.position)
        return {
            'state': self.buffer['state'][indices].astype(np.float32),
            'action': self.buffer['action'][indices].astype(np.int8),
            'reward': self.buffer['reward'][indices].astype(np.float32),
            'next_state': self.buffer['next_state'][indices].astype(np.float32),
            'done': self.buffer['done'][indices].astype(np.float32)
        }
