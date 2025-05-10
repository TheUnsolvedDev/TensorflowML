import numpy as np
import tensorflow as tf
import os

from model import *
from buffer import *
from config import *


class DuelingDoubleDeepQNetworkAgent:
    def __init__(self, input_shape=(IMG_SIZE[0], IMG_SIZE[1], NUM_CHANNELS), num_actions=NUM_ACTIONS, buffer_size=BUFFER_SIZE, num_envs=NUM_ENVS,
                 alpha=ALPHA, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE, num_update_every=NUM_UPDATE_EVERY, num_update_target_every=NUM_UPDATE_TARGET_EVERY):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.buffer_size = buffer_size
        self.num_envs = num_envs

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = num_update_every
        self.update_target_every = num_update_target_every

        self.model = DuelingQModel(input_shape, num_actions)
        self.target_model = DuelingQModel(input_shape, num_actions)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.model.summary()

        self.replay_buffer = ReplayBuffer(buffer_size, num_envs)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05

    @tf.function
    def _act(self, state):
        if state.ndim == 3:
            state = tf.expand_dims(state, axis=0)
        q_values = self.model(state)
        action = tf.argmax(q_values, axis=1)
        return action

    def act(self, state, epsilon=0.01, test=False):
        if np.random.rand() < epsilon:
            if self.num_envs == 1 or test == True:
                return np.random.randint(0, self.num_actions)
            return np.random.randint(0, self.num_actions, size=(self.num_envs,))
        else:
            if self.num_envs == 1 or test == True:
                return self._act(state).numpy()[0]
            return self._act(state).numpy()

    @tf.function
    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values_selected = tf.reduce_sum(
                q_values * tf.one_hot(actions, depth=self.num_actions), axis=1
            )
            q_next_online = self.model(next_states)
            best_actions = tf.argmax(q_next_online, axis=1) 
            q_next_target = self.target_model(next_states)
            max_q_next = tf.reduce_sum(
                q_next_target * tf.one_hot(best_actions, self.num_actions), axis=1
            )
            targets = rewards + (1.0 - dones) * self.gamma * max_q_next
            targets = tf.stop_gradient(targets)
            loss = tf.reduce_mean(tf.square(q_values_selected - targets))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def update_target(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = (1 - self.tau) * target_weights[i] + self.tau * model_weights[i]
        self.target_model.set_weights(target_weights)
        
    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        path = 'models/model_weights.weights.h5'
        self.model.save_weights(path)
        
    def load_model(self):
        path = 'models/model_weights.weights.h5'
        if not os.path.exists(path):
            self.save_model()
        self.model.load_weights(path)
        self.target_model.set_weights(self.model.get_weights())
