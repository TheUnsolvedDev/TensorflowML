import gymnasium as gym
import numpy as np
import collections
import ale_py

from config import *



class VecFrameStack(gym.vector.VectorWrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)

        obs_shape = env.single_observation_space.shape  # (C, H, W)
        self.channels = obs_shape[0]
        self.stack_shape = (self.channels * k, *obs_shape[1:])  # (C * k, H, W)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_envs, *self.stack_shape),
            dtype=env.single_observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs.copy())
        stacked_obs = self._get_obs()

        # Reward shaping
        reward = np.array(reward, dtype=np.float16)
        reward = np.where(reward > 0, 3.0, reward)     # +1 → +3
        reward = np.where(reward < 0, -1.0, reward)    # -1 → -1
        reward += 0.01                                  # survival bonus
        reward += self._ball_motion_reward()            # motion reward
        reward = np.clip(reward, -1.0, 3.5)              # optional

        return stacked_obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Stack frames into (B, C * k, H, W)
        return np.concatenate(list(self.frames), axis=1)

    def _ball_motion_reward(self):
        if len(self.frames) < 2:
            return np.zeros(self.num_envs, dtype=np.float32)

        diff = np.abs(self.frames[-1] - self.frames[-2])

        # Sum over all axes except batch
        axes = tuple(range(1, diff.ndim))
        motion_score = np.sum(diff, axis=axes) / (255.0 * np.prod(diff.shape[1:]))
        return 0.1 * motion_score.astype(np.float32)




class Environment:
    def __init__(self, env_name='PongNoFrameskip-v4',num_envs=NUM_ENVS):
        if num_envs > 1:
            self.env = gym.make_vec(id=env_name,num_envs=num_envs,vectorization_mode='async')
            self.env = gym.wrappers.vector.GrayscaleObservation(self.env)
            self.env = gym.wrappers.vector.ResizeObservation(self.env, IMG_SIZE)
            self.env = VecFrameStack(self.env, NUM_STACK_FRAMES) 
            self.num_envs = num_envs
        else:
            self.env = gym.make(env_name)#, render_mode='human')
            self.env = gym.wrappers.GrayscaleObservation(self.env)
            self.env = gym.wrappers.ResizeObservation(self.env, IMG_SIZE)
            self.env = gym.wrappers.FrameStackObservation(self.env, NUM_STACK_FRAMES)
            self.num_envs = 1
            
        self.state_size = IMG_SIZE
        self.action_size = NUM_ACTIONS
        
    def wrapper(self,env):
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.ResizeObservation(env, IMG_SIZE)
        env = gym.wrappers.FrameStackObservation(env, NUM_STACK_FRAMES)
        return env
    
    def reset(self):
        state, info = self.env.reset()
        if self.num_envs > 1:
            state = np.array(state).reshape(self.num_envs, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
        else:
            state = np.array(state).reshape(1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
        return state
    
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        done = np.logical_or(truncated, done)
        if self.num_envs > 1:
            next_state = np.array(next_state).reshape(self.num_envs, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
        else:
            next_state = np.array(next_state).reshape(1, IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS)
        return next_state, reward, done, info
    
if __name__ == "__main__":
    env = Environment(num_envs=NUM_ENVS)
    state = env.reset()
    done = False
    while not np.all(done):
        action = env.env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        print(f"State: {state.shape}, Action: {action}, Reward: {reward}, Next State: {next_state.shape}")
        state = next_state
    env.env.close()
        
        