import gymnasium as gym
import numpy as np

from config import *

class MountainCarRewardWrapper(gym.vector.VectorRewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_pos = obs[..., 0]  # track position
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Encourage progress toward goal
        pos = obs[..., 0]
        shaped_reward = reward + (pos - self.prev_pos) * 10.0
        self.prev_pos = pos
        return obs, shaped_reward, terminated, truncated, info

class Environment:
    def __init__(self, name=ENV_NAME, num_envs=NUM_ENVS):
        self.name = name
        self.num_envs = num_envs
        if self.num_envs > 1:
            self.envs = gym.make_vec(
                name, num_envs=self.num_envs, vectorization_mode='async')
            self.envs = MountainCarRewardWrapper(self.envs) 
        else:
            self.envs = gym.make(name)#, render_mode='human')
            self.envs = gym.wrappers.Autoreset(self.envs)

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
        actions = np.random.randint(low=0, high=2, size=(env.num_envs,))
        next_state, rewards, dones, info = env.step(actions)
        state = next_state
        if np.any(dones):
            print(dones,state.shape, rewards)


if __name__ == '__main__':
    simulate()