import tensorflow as tf
import numpy as np

from config import *
from model import Lenet5, Lenet5_advantage
from buffer import ReplayBuffer


class QNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = BATCH_SIZE
        self.memory = ReplayBuffer(self.batch_size)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = 0.995
        self.epsilon_min = EPSILON_END
        self.learning_rate = LEARNING_RATE
        self.tau = 0.9
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def update_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def epsilonm_linear_decay(self, start, end, steps, current_step):
        final = start - (start - end) * current_step / steps
        return max(final, end)

    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        return self.epsilon

    @tf.function
    def predict(self, state):
        state = tf.expand_dims(state, axis=0)
        return self.model(state)

    def get_action(self, state, epsilon=0):
        if tf.random.uniform(()) < epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return tf.argmax(self.predict(state))[0].numpy()


class DQNAgent(QNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = Lenet5(self.state_size, self.action_size)
        self.target_model = Lenet5(self.state_size, self.action_size)
        self.update_weights()

    @tf.function
    def train_one_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.action_size)
            q_values = tf.reduce_sum(predictions * one_hot_actions, axis=1)
            next_predictions = self.target_model(next_states)
            next_q_values = tf.reduce_max(next_predictions, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + \
                (1 - dones) * self.gamma * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss


class DuelingDQNAgent(QNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = Lenet5_advantage(self.state_size, self.action_size)
        self.target_model = Lenet5_advantage(self.state_size, self.action_size)
        self.update_weights()

    @tf.function
    def train_one_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.action_size)
            q_values = tf.reduce_sum(predictions * one_hot_actions, axis=1)
            next_predictions = self.target_model(next_states)
            next_q_values = tf.reduce_max(next_predictions, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + \
                (1 - dones) * self.gamma * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss


class DoubleDQNAgent(QNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = Lenet5(self.state_size, self.action_size)
        self.target_model = Lenet5(self.state_size, self.action_size)
        self.update_weights()

    @tf.function
    def train_one_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.action_size)
            q_values = tf.reduce_sum(predictions * one_hot_actions, axis=1)
            next_predictions = self.model(next_states)
            next_actions = tf.argmax(next_predictions, axis=1)
            next_actions_one_hot = tf.one_hot(next_actions, self.action_size)
            target_next_predictions = self.target_model(next_states)
            next_q_values = tf.reduce_sum(
                target_next_predictions * next_actions_one_hot, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + \
                (1 - dones) * self.gamma * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss


class DoubleDuelingDQNAgent(QNAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = Lenet5_advantage(self.state_size, self.action_size)
        self.target_model = Lenet5_advantage(self.state_size, self.action_size)
        self.update_weights()

    @tf.function
    def train_one_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.action_size)
            q_values = tf.reduce_sum(predictions * one_hot_actions, axis=1)
            next_predictions = self.model(next_states)
            next_actions = tf.argmax(next_predictions, axis=1)
            next_actions_one_hot = tf.one_hot(next_actions, self.action_size)
            target_next_predictions = self.target_model(next_states)
            next_q_values = tf.reduce_sum(
                target_next_predictions * next_actions_one_hot, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + \
                (1 - dones) * self.gamma * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss
