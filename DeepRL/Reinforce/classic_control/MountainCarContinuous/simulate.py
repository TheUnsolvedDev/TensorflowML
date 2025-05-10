import gymnasium
import tensorflow as tf
import numpy as np

from config import *
from agent import *
from environment import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def simulate(steps=1000):
    env = gym.make(ENV_NAME, render_mode='human')
    agent = ReinforceAgent(input_shape=ENV_INPUT_SHAPE,
                               num_envs=1)
    agent.load_model()
    done = False
    state = env.reset()[0]
    rewards = 0
    while not done:
        action = agent.act(state, test=True)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        rewards += reward
        if done or truncated:
            print(f"Episode finished after {steps} steps with reward: {rewards}")
            break
    env.close()
    print("Simulation complete")
    
if __name__ == '__main__':
    simulate()