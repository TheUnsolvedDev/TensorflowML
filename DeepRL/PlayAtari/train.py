import tensorflow as tf
import numpy as np
import tqdm
import gc
import argparse

from config import *
from dqn_agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent,DoubleDuelingDQNAgent
from environment import AtariGame



def history_save(array, file="history.txt"):
    with open(file, "a") as f:
        for element in array:
            f.write(str(element) + ",")
        f.write("\n")

class TrainingSim:
    def __init__(self,type = 'vanilla',agent = None):
        self.env = AtariGame(ENV,render_mode=None)
        self.action_space = self.env.action_space
        self.agent_type = type
        self.agent = agent
        self.summary_writer = tf.summary.create_file_writer(f'{LOGS_PATH}{self.agent_type}')

        
    def train(self):
        counter = 0
        max_reward = -np.inf
        epsilon = self.agent.epsilonm_linear_decay(start=EPSILON_START, end=EPSILON_END, steps=NUM_DECAY_STEPS, current_step=0)
        for i in tqdm.tqdm(range(1,NUM_GAMES+1)):
            state = self.env.reset()
            done = False
            episodic_reward = 0
            max_reward = max(max_reward, episodic_reward)
            while not done:
                action = self.agent.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                self.agent.memory.add(state, action, reward, next_state, done)
                episodic_reward += reward
                if self.agent.memory.position >= self.agent.batch_size:
                    epsilon = self.agent.epsilonm_linear_decay(start=EPSILON_START, end=EPSILON_END, steps=NUM_DECAY_STEPS,current_step=i)
                    batch = self.agent.memory.get()
                    loss = self.agent.train_one_step(batch['state'], batch['action'], batch['reward'], batch['next_state'], batch['done'])
                    gc.collect()
                    counter += 1
                    if counter % UPDATE_TARGET == 0:
                        self.agent.update_weights()
                        if episodic_reward >= max_reward:
                            self.agent.model.save_weights(f"{MODEL_PATH}{self.agent_type}.weights.h5")
                state = next_state
            with self.summary_writer.as_default(step=i):
                tf.summary.scalar("reward", episodic_reward, step=i)
                tf.summary.scalar("epsilon", epsilon, step=i)
                tf.summary.scalar("loss", loss, step=i)
            # history_save([i, episodic_reward, epsilon, loss.numpy()], file="history_"+self.agent_type+".txt")
            print(f"Episode: {i}, Reward: {episodic_reward}, Epsilon: {epsilon:.4f}, Loss: {loss}")
            # gc.collect()
                
if __name__ == "__main__":
    sim = TrainingSim()
    sim.train()
            