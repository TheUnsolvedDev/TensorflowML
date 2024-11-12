import tensorflow as tf
import numpy as np
import argparse
import gymnasium as gym
import tqdm

from config import *
from environment import AtariGame
from dqn_agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, DoubleDuelingDQNAgent
from train import TrainingSim

tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices('GPU')[0], True)


def main():
    env = AtariGame(ENV, render_mode=None)
    test_env = AtariGame(ENV, render_mode=None)
    
    train_log_summary = tf.summary.create_file_writer(f'{LOGS_PATH}train')
    test_log_summary = tf.summary.create_file_writer(f'{LOGS_PATH}test')

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='vanilla',
                        help='type of agent', choices=['vanilla', 'dueling', 'double', 'hyper'])
    args = parser.parse_args()
    if args.type == 'vanilla':
        agent = DQNAgent([STATE_SHAPE[0], STATE_SHAPE[1],
                         STACK_FRAMES], env.action_space.n)
    elif args.type == 'dueling':
        agent = DuelingDQNAgent(
            [STATE_SHAPE[0], STATE_SHAPE[1], STACK_FRAMES], env.action_space.n)
    elif args.type == 'double':
        agent = DoubleDQNAgent(
            [STATE_SHAPE[0], STATE_SHAPE[1], STACK_FRAMES], env.action_space.n)
    elif args.type == 'hyper':
        agent = DoubleDuelingDQNAgent(
            [STATE_SHAPE[0], STATE_SHAPE[1], STACK_FRAMES], env.action_space.n)

    state = env.reset()
    epsilon = agent.epsilonm_linear_decay(
        EPSILON_START, EPSILON_END, NUM_DECAY_STEPS, 0)
    max_reward = -np.inf
    episode_reward = 0
    games = 0
    for step in tqdm.tqdm(range(1, NUM_STEPS+1)):
        action = agent.get_action(state=state, epsilon=epsilon)
        next_state, reward, done, info = env.step(action)
        agent.memory.add(state,action,reward,next_state,done)
        episode_reward += reward
        if agent.memory.position >= agent.batch_size:
            epsilon = agent.epsilonm_linear_decay(
                EPSILON_START, EPSILON_END, NUM_DECAY_STEPS, step)
            batch = agent.memory.get()
            loss = agent.train_one_step(batch['state'], batch['action'], batch['reward'], batch['next_state'], batch['done'])
            if step % UPDATE_TARGET == 0:
                agent.update_weights()
                if episode_reward >= max_reward:
                    agent.model.save_weights(f"models/{args.type}.weights.h5")
        state = next_state
        if done:
            games += 1
            state = env.reset()
            max_reward = max(max_reward, episode_reward)
            with train_log_summary.as_default(step=step):
                tf.summary.scalar("reward", episode_reward, step=step)
                tf.summary.scalar("epsilon", epsilon, step=step)
                tf.summary.scalar("loss", loss, step=step)
            episode_reward = 0
            
        if step % TEST_EVERY == 0:
            test_env = AtariGame(ENV,render_mode='human')
            state = test_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.get_action(state=state, epsilon=0.05)
                next_state, reward, done, info = test_env.step(action)
                episode_reward += reward
                state = next_state
            with test_log_summary.as_default(step=step):
                tf.summary.scalar("reward", episode_reward, step=step)
            
            test_env.reset()
            # test_env.viewer = None
            test_env.env.close()

    env.env.close()
    # agent.load_weights(f"model/{args.type}.weights.h5")

    # sim = TrainingSim(type=args.type, agent=agent)
    # sim.train()


if __name__ == '__main__':
    main()
