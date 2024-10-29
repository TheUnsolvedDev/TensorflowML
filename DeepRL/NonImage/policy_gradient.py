import tensorflow as tf
import numpy as np
import gymnasium as gym
import gc
import matplotlib.pyplot as plt


def policy_model(input_shape, output_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


class PolicyGradientAgent:
    def __init__(self, state_shape, action_shape, learning_rate=0.0001):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.model = policy_model(self.state_shape, self.action_shape)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

    @tf.function
    def update(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            probs = self.model(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_shape), axis=1)
            loss = -tf.reduce_mean(tf.math.log(action_probs) * rewards)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def predict(self, state):
        probs = self.model(state)
        return probs


class TrajectoryBuffer:
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        self.states = np.array(self.states, dtype=np.float32)
        self.states = self.states.reshape(self.states.shape[0], -1)
        self.actions = np.array(self.actions, dtype=np.int32)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        return self.states, self.actions, self.rewards


def train_agent(episodes=1000, max_steps=500):
    env = gym.make('CartPole-v1')
    print(env.observation_space.shape, env.action_space.n)
    agent = PolicyGradientAgent(
        state_shape=env.observation_space.shape, action_shape=env.action_space.n)
    buffer = TrajectoryBuffer()
    
    episodic_rewards = []
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        for t in range(max_steps):
            state = np.expand_dims(state, axis=0)
            probs = agent.predict(state)[0]
            action = np.random.choice(len(probs), p=probs.numpy())
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            buffer.add(state, action, reward, done)
            state = next_state
            if done:
                break
        episodic_rewards.append(total_reward)
        tf.keras.backend.clear_session()
        gc.collect()
        states, actions, rewards = buffer.get()
        loss = agent.update(states, actions, rewards)
        buffer.clear()
        if episode % 100 == 0:
            print(f'Episode: {episode}, Reward: {total_reward}', 'Loss:', loss.numpy())
    return rewards


if __name__ == '__main__':
    train_agent()
