import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import sys
import gc
import os

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
        
def train(environment, agent, step, writer, num_steps=1000):
    state = environment.reset()
    for _ in range(0, num_steps):
        current_step = _ + step - 1
        action = agent.act(state)
        next_state, reward, done, info = environment.step(action)
        loss = agent.learn(state, action, reward, next_state, done)
        with writer.as_default():
            tf.summary.scalar('loss', loss, step=current_step)
        print(
            f"\rTraining step: [{_}/{num_steps}] \tLoss: {loss.numpy():.4f}", end='')
        sys.stdout.flush()
        state = next_state
    return agent

def test(environment, agent, num_steps=int(1e+4)):
    state = environment.reset()
    done = False
    rewards_collected = 0
    for _ in range(num_steps):
        actions = agent.act(state, test=True)
        next_state, rewards, done, info = environment.step(actions)
        state = next_state
        rewards_collected += rewards
        if done:
            break
    return rewards_collected

def running_average(rewards, num_steps=100):
    avg_rewards = np.zeros(len(rewards))
    for i in range(len(rewards)):
        if i < num_steps:
            avg_rewards[i] = np.mean(rewards[:i+1])
        else:
            avg_rewards[i] = np.mean(rewards[i-num_steps:i+1])
    return avg_rewards

def plot_rewards(avg_reward_collection):
    plt.plot(avg_reward_collection)
    plt.plot(running_average(avg_reward_collection, num_steps=100))
    plt.title("Average Test Reward")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Test Reward")
    plt.grid()
    plt.savefig('rewards/avg_test_reward.png')
    plt.close()

def session(num_steps=NUM_TRAINING_STEPS):
    env = Environment(num_envs=NUM_ENVS)
    test_env = Environment(num_envs=1)
    agent = ActorCriticAgent(input_shape=ENV_INPUT_SHAPE,
                               output_shape=ENV_OUTPUT_SHAPE,
                               num_envs=NUM_ENVS,
                               learning_rate=ALPHA,
                               gamma=GAMMA)
    num_steps_to_train_for = int(1e+3)

    os.makedirs('rewards', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    logdir = f"logs/train_logs"
    writer = tf.summary.create_file_writer(logdir)

    maxm_reward = -np.inf
    agent.load_model()
    avg_reward_collection = []

    for _ in tqdm.tqdm(range(1, num_steps+1, num_steps_to_train_for)):
        agent = train(environment=env, step=_, agent=agent,
                      writer=writer, num_steps=num_steps_to_train_for)

        test_rewards = [test(environment=test_env, agent=agent) for i in range(10)]
        avg_test_reward = np.mean(test_rewards)

        with writer.as_default():
            tf.summary.scalar("Average Test Reward", avg_test_reward, step=_)

        avg_reward_collection.append(avg_test_reward)
        print(f"\nAverage test reward: {avg_test_reward:.2f}")

        gc.collect()
        tf.keras.backend.clear_session()

        if avg_test_reward > maxm_reward:
            maxm_reward = avg_test_reward
            agent.save_model()
            print(f"New best model saved with reward: {maxm_reward:.2f}")

    return avg_reward_collection

if __name__ == "__main__":
    plot_rewards(session())