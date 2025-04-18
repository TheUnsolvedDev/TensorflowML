import tensorflow as tf
import numpy as np

from environment import *
from config import *
from agent import *

def test(environment, agent, num_steps=int(1e+4)):
    state = environment.reset()
    done = False
    rewards_collected = 0
    for _ in range(num_steps):
        actions = agent.act(state, epsilon=0.05, test=True)
        next_state, rewards, done, info = environment.step(actions)
        state = next_state
        rewards_collected += rewards
        if done:
            break
    return rewards_collected

if __name__ == '__main__':
    agent = DeepQNetworkAgent(input_shape=(IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS), num_actions=NUM_ACTIONS, buffer_size=BUFFER_SIZE, num_envs=NUM_ENVS,
                             alpha=ALPHA, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE, num_update_every=NUM_UPDATE_EVERY, num_update_target_every=NUM_UPDATE_TARGET_EVERY)
    environment = Environment(num_envs=1)
    agent.load_model()
    rewards = test(environment, agent, num_steps=int(1e+4))
    print(f"Test rewards: {rewards}")