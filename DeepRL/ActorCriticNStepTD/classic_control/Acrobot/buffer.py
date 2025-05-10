import numpy as np

from config import *


class TrajectoryBuffer:
    def __init__(self, num_envs, step=N_STEP):
        self.step = step
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        self.buffer = {i: [] for i in range(self.num_envs)}
        self.pointers = np.zeros(self.num_envs, dtype=int)

    def add(self, obs, act, rew, next_obs, done):
        for i in range(self.num_envs):
            self.buffer[i].append(
                (obs[i], act[i], rew[i], next_obs[i], done[i]))
            self.pointers[i] += 1

    def pop_trajectories_n_step(self, index, step=N_STEP):
        traj = self.buffer[index]
        collected = []
        for t in traj:
            collected.append(t)
            if len(collected) == step or t[4]:  # done
                break
        self.clear(index)
        states, actions, rewards, next_states, dones = zip(*collected)
        final_next_state = next_states[-1]
        final_done = dones[-1]
        return (
            np.array(states), np.array(actions), np.array(rewards),
            final_next_state, final_done
        )
        

    def clear(self, index):
        if index in self.buffer:
            self.buffer[index] = []
            self.pointers[index] = 0
