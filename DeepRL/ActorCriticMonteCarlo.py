import silence_tensorflow.auto
import tqdm
import tensorflow as tf
import functools
import numpy as np
import tensorflow_probability as tfp
import gymnasium as gym
import os
import gc
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('NonAtari/ActorCriticMonteCarlo/Cartpole')

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate


class PolicyNetwork:
    def __init__(self, input_shape: list, output_shape: int) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.source_model = self.get_model()

    def get_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        outputs = tf.keras.layers.Dense(
            self.output_shape, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)


class ValueNetwork:
    def __init__(self, input_shape: list) -> None:
        self.input_shape = input_shape
        self.source_model = self.get_model()

    def get_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)  # Output single value
        return tf.keras.Model(inputs, outputs)


class Buffer:
    def __init__(self) -> None:
        self.states = None
        self.actions = None
        self.rewards = None
        self.reset()

    def reset(self):
        del self.states
        del self.actions
        del self.rewards
        self.states, self.actions, self.rewards = [], [], []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def sample(self):
        states = np.array(self.states).reshape(-1, 4).astype(np.float32)
        actions = np.array(self.actions).reshape(-1, )
        rewards = np.array(self.rewards).reshape(-1, ).astype(np.float32)
        self.reset()
        return states, actions, rewards


class Policy:
    @staticmethod
    def action_value(model, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        @tf.function
        def call_model(model, state):
            return model(state, training=False)

        probs = call_model(model, state)
        action = tf.random.categorical(probs, 1)
        return tf.squeeze(action, axis=-1).numpy()[0]


class AC_Agent:
    def __init__(self):
        self.actor = PolicyNetwork(input_shape=(4,), output_shape=2)
        self.critic = ValueNetwork(input_shape=(4,))
        self.actor_source = self.actor.source_model
        self.critic_source = self.critic.source_model
        self.gamma = GAMMA
        self.alpha = LR
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.alpha)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.alpha)

    def act(self, state):
        return Policy.action_value(self.actor_source, state)

    def discount_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    @tf.function
    def update(self, states, actions, discounted_rewards):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            actor_tape.watch(self.actor_source.trainable_variables)
            critic_tape.watch(self.critic_source.trainable_variables)

            action_prob = self.actor_source(states)
            selected_action_prob = tf.reduce_sum(
                action_prob * tf.one_hot(actions, depth=2), axis=1)
            critic_value = tf.squeeze(self.critic_source(states), axis=-1)

            actor_loss = - \
                tf.reduce_mean(tf.math.log(selected_action_prob)
                               * discounted_rewards)
            critic_loss = tf.reduce_mean(
                tf.square(discounted_rewards - critic_value))

        actor_gradients = actor_tape.gradient(
            actor_loss, self.actor_source.trainable_variables)
        critic_gradients = critic_tape.gradient(
            critic_loss, self.critic_source.trainable_variables)

        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor_source.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic_source.trainable_variables))

        return actor_loss, critic_loss


class TrainEnv:
    def __init__(self, name: str):
        self.env = gym.make(name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class TestEnv:
    def __init__(self, name: str):
        self.env = gym.make(name, render_mode='rgb_array')
        self.env = gym.wrappers.RecordVideo(
            self.env, video_folder='NonAtari/ActorCriticMonteCarlo/Cartpole', episode_trigger=lambda x: True)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class Environment:
    def __init__(self):
        self.env_name = 'CartPole-v1'
        self.train = TrainEnv(self.env_name)
        self.test = TestEnv(self.env_name)

    def get_environments(self):
        return self.train, self.test


def simulate(num_games=500, num_episodes=1000):
    env = Environment()
    agent = AC_Agent()
    buffer = Buffer()

    train, test = env.get_environments()
    actor_loss, critic_loss = 0.00, 0.00
    state, obs = train.reset()
    for game in tqdm.tqdm(range(num_games*num_episodes)):
        state = np.expand_dims(state, axis=0)
        action = agent.act(state)
        next_state, reward, done, truncated, info = train.step(action)
        buffer.store(state, action, reward)
        state = next_state
        if done:
            states, actions, rewards = buffer.sample()
            discounted_rewards = agent.discount_rewards(rewards, agent.gamma)
            actor_loss, critic_loss = agent.update(
                states, actions, discounted_rewards)
            writer.add_scalar("Actor_Loss/train", float(actor_loss), game)
            writer.add_scalar("Critic_Loss/train", float(critic_loss), game)
            state, obs = train.reset()
            gc.collect()

        if game % num_episodes == 0:
            test_state = test.reset()[0]
            rewards_history = 0
            for i in range(num_episodes):
                test_state = np.expand_dims(test_state, axis=0)
                action = agent.act(test_state)
                test_next_state, rewards, done, truncated, info = test.step(
                    action)
                rewards_history += rewards
                test_state = test_next_state
                if done or truncated:
                    break
            writer.add_scalar("Rewards/test", rewards_history, game)
            gc.collect()


if __name__ == '__main__':
    simulate()
