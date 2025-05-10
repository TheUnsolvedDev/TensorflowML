import gymnasium as gym
import numpy as np

from config import *

class Environment:
    def __init__(self, name=ENV_NAME, num_envs=NUM_ENVS):
        self.name = name
        self.num_envs = num_envs
        if self.num_envs > 1:
            self.envs = gym.make_vec(
                name, num_envs=self.num_envs, vectorization_mode='async',continuous=False)
            self.envs = gym.wrappers.vector.GrayscaleObservation(self.envs)
            self.envs = gym.wrappers.vector.ResizeObservation(self.envs,(ENV_INPUT_SHAPE[0],ENV_INPUT_SHAPE[1]))
            self.envs = gym.wrappers.vector.ReshapeObservation(self.envs,(ENV_INPUT_SHAPE[0],ENV_INPUT_SHAPE[1],1))
        else:
            self.envs = gym.make(name,continuous=False)#, render_mode='human')
            self.envs = gym.wrappers.Autoreset(self.envs)
            self.envs = gym.wrappers.GrayscaleObservation(self.envs)
            self.envs = gym.wrappers.ResizeObservation(self.envs,(ENV_INPUT_SHAPE[0],ENV_INPUT_SHAPE[1]))
            self.envs = gym.wrappers.ReshapeObservation(self.envs,(ENV_INPUT_SHAPE[0],ENV_INPUT_SHAPE[1],1))

    def reset(self):
        return self.envs.reset()[0]

    def step(self, actions):
        next_state, rewards, dones, truncated, info = self.envs.step(actions)
        dones = np.logical_or(dones, truncated)
        return next_state, rewards, dones, info


def simulate(steps=1000):
    env = Environment(num_envs=NUM_ENVS)
    state = env.reset()
    for _ in range(steps):
        actions = np.random.randint(low=0, high=ENV_OUTPUT_SHAPE[0], size=(env.num_envs,))
        next_state, rewards, dones, info = env.step(actions)
        state = next_state
        # if np.any(dones):
        print(dones,state.shape)


if __name__ == '__main__':
    simulate()