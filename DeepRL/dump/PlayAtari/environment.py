import gymnasium as gym
import numpy as np
import ale_py

from config import *
from wrappers import *
gym.register_envs(ale_py)


def visualize_state(state):
    state = state.reshape(STACK_FRAMES, STATE_SHAPE[0], STATE_SHAPE[1])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        ax[i//2, i % 2].imshow(state[i])
    plt.show()


class AtariGame:
    def __init__(self, env_name, render_mode=None):
        if render_mode is not None:
            self.env = gym.make(env_name, render_mode=render_mode)
        else:
            self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.additives()

    def additives(self):
        self.env = gym.wrappers.GrayscaleObservation(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env, (STATE_SHAPE[0], STATE_SHAPE[1]))
        self.env = gym.wrappers.FrameStackObservation(self.env, STACK_FRAMES)
        self.env = gym.wrappers.ReshapeObservation(self.env, (STATE_SHAPE[0], STATE_SHAPE[1], STACK_FRAMES))

    def step(self, action):
        state, reward, done,truncated, info = self.env.step(action)
        return state, reward, done or truncated , info

    def reset(self):
        state = self.env.reset(seed=0)
        return state[0]


if __name__ == "__main__":
    env = AtariGame(ENV)
    state = env.reset()
    print(state.shape)
    visualize_state(state)
    for i in range(10000):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(reward)
        if done:
            state = env.reset()[0]
        visualize_state(state)
