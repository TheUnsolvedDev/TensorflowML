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
writer = SummaryWriter('NonAtari/DoubleDeepQNetwork/Cartpole')

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

BUFFER_SIZE = 4*int(1e4)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
# how often to update the network (When Q target is present)
UPDATE_EVERY = 20


class QNetwork:
    def __init__(self, input_shape: list, output_shape: int) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.source_model = self.get_model()
        self.target_model = self.get_model()

    def update_weights(self, tau: float = 0.9) -> None:
        for source_layer, target_layer in zip(self.source_model.layers, self.target_model.layers):
            if isinstance(source_layer, tf.keras.layers.Dense):
                source_weights, source_biases = source_layer.get_weights()
                target_weights, target_biases = target_layer.get_weights()
                updated_weights = target_weights + tau * \
                    (source_weights - target_weights)
                updated_biases = target_biases + tau * \
                    (source_biases - target_biases)
                target_layer.set_weights([updated_weights, updated_biases])

    def get_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        outputs = tf.keras.layers.Dense(
            self.output_shape, activation='linear')(x)
        return tf.keras.Model(inputs, outputs)


class ReplayBuffer:
    def __init__(self, queue_capacity=BUFFER_SIZE) -> None:
        self.capacity = queue_capacity
        self.max_len = 0
        self.storage = []

    def store(self, state, action, reward, next_state, done):
        mdp_tuple = (state, action, reward, next_state, done)
        if len(self.storage) > self.capacity:
            self.max_len -= 1
            self.storage.pop(0)
        self.max_len += 1
        self.storage.append(mdp_tuple)

    def sample(self, batch_size=BATCH_SIZE):
        indices = np.random.randint(0, self.max_len, size=batch_size)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for ind in indices:
            states.append(self.storage[ind][0])
            actions.append(self.storage[ind][1])
            rewards.append(self.storage[ind][2])
            next_states.append(self.storage[ind][3])
            dones.append(self.storage[ind][4])

        states = np.array(states).reshape(-1, 4).astype(np.float32)
        actions = np.array(actions).reshape(-1, )
        rewards = np.array(rewards).reshape(-1, 1).astype(np.float32)
        next_states = np.array(next_states).reshape(-1, 4).astype(np.float32)
        dones = np.array(dones).reshape(-1, 1).astype(np.float32)
        return states, actions, rewards, dones, next_states


class Policy:

    @staticmethod
    def epsilon_greedy(model, state, epsilon=0.1):
        state = np.array(state)

        @tf.function
        def call_model(model, state):
            return model(state)

        q_values = call_model(model, state)
        if np.random.rand() < epsilon:
            return np.random.randint(2, size=(state.shape[0]))
        else:
            return np.argmax(q_values, axis=1)

    @staticmethod
    def softmax(model, state, tau=0.5):
        state = np.array(state)

        @tf.function
        def call_model(model, state):
            return model(state)

        q_values = call_model(model, state)
        probs = tf.nn.softmax(q_values/tau).numpy()
        action = tf.random.categorical(probs, 1)
        return tf.squeeze(action, axis=-1)


class DoubleDQN_Agent:
    def __init__(self, strategy='epsilon_greedy'):
        self.network = QNetwork(input_shape=(4,), output_shape=2)
        self.source = self.network.source_model
        self.target = self.network.target_model
        self.network.update_weights()
        self.strategy = strategy
        self.alpha = LR
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.alpha)
        self.gamma = GAMMA

    def act(self, state, param=0.1):
        if self.strategy == 'epsilon_greedy':
            return Policy.epsilon_greedy(self.source, state, param)
        elif self.strategy == 'softmax':
            return Policy.softmax(self.source, state, param)

    @tf.function
    def update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            tape.watch(self.source.trainable_weights)
            next_actions = tf.one_hot(
                tf.argmax(self.source(next_states), 1), 2)
            target = tf.stop_gradient(self.target(next_states))
            target = tf.reduce_sum((1.0-dones)*target*next_actions, axis=1)
            current = self.source(states)
            current = tf.reduce_sum(current*tf.one_hot(actions, 2), axis=1)
            loss = self.mse_loss(
                rewards + self.gamma*target, current)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.source.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.source.trainable_weights))

        return loss


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
            self.env, video_folder='NonAtari/DoubleDeepQNetwork/Cartpole', episode_trigger=lambda x: True)

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


def linear_annealing_epsilon(initial_epsilon, min_epsilon, annealing_steps, t):
    if t >= annealing_steps:
        return min_epsilon
    else:
        return max(initial_epsilon - (t / annealing_steps) * (initial_epsilon - min_epsilon), min_epsilon)


def simulate(num_games=1000, num_episodes=1000):
    env = Environment()
    agent = DoubleDQN_Agent()
    buffer = ReplayBuffer()
    start_epsilon, end_epsilon = 1, 0.01
    epsilon = start_epsilon

    train, test = env.get_environments()
    agent.network.update_weights()

    loss = 0.00
    state, obs = train.reset()
    for game in tqdm.tqdm(range(num_games*num_episodes)):
        state = np.expand_dims(state, axis=0)
        action = agent.act(state, epsilon)[0]
        next_state, reward, done, truncated, info = train.step(action)
        buffer.store(state, action, reward, next_state, done)
        epsilon = linear_annealing_epsilon(
            start_epsilon, end_epsilon, num_games*num_episodes//2, game)
        state = next_state

        if done:
            state, obs = train.reset()

        if buffer.max_len > 128:
            states, actions, rewards, dones, next_states = buffer.sample()
            loss = agent.update(
                states, actions, rewards, next_states, dones)
            writer.add_scalar("Loss/train", float(loss), game)
            if game % UPDATE_EVERY == 0:
                agent.network.update_weights()

        if game % num_episodes == 0:
            test_state = test.reset()[0]
            rewards_history = 0
            for i in range(num_episodes):
                test_state = np.expand_dims(test_state, axis=0)
                action = agent.act(test_state, 0.01)[0]
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
