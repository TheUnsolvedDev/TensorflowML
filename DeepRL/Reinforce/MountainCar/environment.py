import gymnasium as gym
import numpy as np

from config import *

class MountainCarRewardShapingVec(gym.vector.VectorWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.goal_position = 0.5

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        positions = obs[:, 0]
        velocities = obs[:, 1]

        # Potential shaping: reward progress
        potential = (positions - (-1.2)) / (self.goal_position - (-1.2))  # normalized [0, 1]
        shaped_rewards = rewards + 1.5 * potential + 0.5 * np.abs(velocities)

        # Optional: Add bonus for solving
        solved = positions >= self.goal_position
        shaped_rewards += solved * 10.0

        return obs, shaped_rewards, terminated, truncated, infos


class Environment:
    def __init__(self, name=ENV_NAME, num_envs=NUM_ENVS):
        self.name = name
        self.num_envs = num_envs
        if self.num_envs > 1:
            self.envs = gym.make_vec(
                name, num_envs=self.num_envs, vectorization_mode='async')
            self.envs = MountainCarRewardShapingVec(self.envs)
        else:
            self.envs = gym.make(name)#, render_mode='human')
            # self.envs = MountainCarRewardShaping(self.envs)
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