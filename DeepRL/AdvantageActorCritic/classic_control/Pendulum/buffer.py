import numpy as np

from config import *

# class TrajectoryBuffer:
#     def __init__(self, num_envs=NUM_ENVS):
#         self.num_envs = num_envs
#         self.buffer = [[] for _ in range(num_envs)]

#     def add(self, state, action, reward):
#         if self.num_envs == 1:
#             self.buffer[0].append((state, action, reward))
#             return self.buffer
#         for i in range(self.num_envs):
#             self.buffer[i].append((state[i], action[i], reward[i]))
#         return self.buffer

#     def get_buffer(self, env_id):
#         trajectory = self.buffer[env_id]
#         states, actions, rewards = zip(*trajectory)
#         states = np.array(states)
#         actions = np.array(actions)
#         rewards = np.array(rewards)
#         return states, actions, rewards

#     def clear(self, env_id):
#         self.buffer[env_id] = []
        
#     def reset(self):
#         for i in range(self.num_envs):
#             self.clear(i)
            
class TrajectoryBuffer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.reset()
        
    def reset(self):
        self.buffer = {i: [[]] for i in range(self.num_envs)}  
        self.pointers = {i: 0 for i in range(self.num_envs)}   

    def add(self, obs, act, rew, done):
        for i in range(self.num_envs):
            # Ensure the current episode list exists at the pointer position
            if self.pointers[i] >= len(self.buffer[i]):
                self.buffer[i].append([])  # Create a new episode list if not exists

            self.buffer[i][self.pointers[i]].append((obs[i], act[i], rew[i]))
            if done[i]:
                self.pointers[i] += 1  # Move to the next episode after done

    def pop_first_trajectories(self):
        trajectories = []
        for i in range(self.num_envs):
            if len(self.buffer[i]) > 0:
                traj = self.buffer[i].pop(0)  # Pop the first trajectory
                self.pointers[i] = max(self.pointers[i] - 1, 0)  # Prevent pointer from going below 0
                trajectories.append(traj)
            else:
                trajectories.append([])  # Empty trajectory if no valid episodes
        return trajectories
    
    def trajectories_to_numpy(self, trajectories):
        # Convert the buffer to numpy arrays
        trajectories_reshapes = []
        for traj in trajectories:
            if len(traj) > 0:
                states, actions, rewards = zip(*traj)
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                trajectories_reshapes.append((states, actions, rewards))
            else:
                trajectories_reshapes.append(([], [], []))
        return trajectories_reshapes
        


def test_trajectory_buffer():
    num_envs = 8
    num_steps = 100
    buffer = TrajectoryBuffer(num_envs=num_envs)

    done_flag = np.zeros(num_envs, dtype=bool)
    for step in range(num_steps):
        obs = np.random.randn(num_envs, 4)      # dummy observation
        act = np.random.randint(0, 2, size=(num_envs,))  # dummy actions
        rew = np.random.randn(num_envs)         # dummy rewards
        done = np.random.rand(num_envs) < 0.1    # ~10% chance to finish an episode

        for i in range(num_envs):
            if done[i]:
                done_flag[i] = True

        if np.all(done_flag):  # If all environments are done
            done_flag = np.zeros(num_envs, dtype=bool)  # Reset done flag
            trajectories = buffer.pop_first_trajectories()  # Pop trajectories
            for i, traj in enumerate(trajectories):
                print(f"Env {i} - Trajectory length: {len(traj)}")
                print('Num Trajectories:', len(buffer.buffer[i]))
                print('Num Pointers:', buffer.pointers[i])
            input()  # Pause to inspect the results

        buffer.insert(obs, act, rew, done)


if __name__ == "__main__":
    test_trajectory_buffer()


                    