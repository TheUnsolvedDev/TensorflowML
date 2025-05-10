import numpy as np
import tensorflow as tf
import os

from config import *
from model import *
from buffer import *


class DuelingDeepQAgent:
    def __init__(self, input_shape=ENV_INPUT_SHAPE, output_shape=ENV_OUTPUT_SHAPE, num_envs=NUM_ENVS, learning_rate=ALPHA, gamma=GAMMA, buffer_size=BUFFER_SIZE, tau=TAU):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size

        self.q_network = QModel(input_shape, output_shape)
        self.target_network = QModel(input_shape, output_shape)
        self.replay_buffer = ReplayBuffer(buffer_size, num_envs)
        self.q_network.summary()
        self.target_network.summary()
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def _act(self, state):
        if state.ndim == 3:
            state = tf.expand_dims(state, axis=0)
        values = self.q_network(state)
        action = tf.argmax(values, axis=1)
        return action

    def act(self, state, epsilon=None, test=False):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() < epsilon:
            if self.num_envs == 1 or test == True:
                return np.random.randint(low=0, high=self.output_shape[0])
            return np.random.randint(low=0, high=self.output_shape, size=(self.num_envs,))
        else:
            if self.num_envs == 1 or test == True:
                return self._act(state).numpy()[0]
            return self._act(state).numpy()

    @tf.function
    def learn(self, states, actions, rewards, next_states, dones):
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            q_values_next = self.target_network(next_states)
            q_values_next_max = tf.reduce_max(q_values_next, axis=1)
            targets = rewards + (1 - dones) * self.gamma * q_values_next_max
            targets = tf.stop_gradient(targets)
            q_values = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.output_shape[0]), axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_values))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        grads = [(tf.clip_by_value(grad, -1, 1)) for grad in grads]
        self.optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables))
        return loss

    def update_target(self):
        model_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = (1 - self.tau) * \
                target_weights[i] + self.tau * model_weights[i]
        self.target_network.set_weights(target_weights)

    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        path = 'models/model_weights.weights.h5'
        self.q_network.save_weights(path)

    def load_model(self):
        path = 'models/model_weights.weights.h5'
        if not os.path.exists(path):
            self.save_model()
        self.q_network.load_weights(path)
        self.target_network.set_weights(self.q_network.get_weights())
