import tensorflow as tf
import numpy as np
import os

from model import *
from config import *
from buffer import *

class PolicyGradientAgent:
    def __init__(self,input_shape=ENV_INPUT_SHAPE,output_shape=ENV_OUTPUT_SHAPE,num_envs=NUM_ENVS,learning_rate=ALPHA,gamma=GAMMA):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.model = PolicyModel(input_shape=self.input_shape,output_shape=self.output_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.trajectory_buffer = TrajectoryBuffer(num_envs=self.num_envs)

    @tf.function
    def _act(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        probs = self.model(state)
        return probs

    def act(self, state, test=False):
        probs = self._act(state).numpy()  # Convert to NumPy
        if self.num_envs == 1 or test:
            action = np.random.choice(probs.shape[-1], p=probs[0])
            return action
        else:
            actions = [np.random.choice(p.shape[-1], p=p) for p in probs]
            return np.array(actions)
    
    @tf.function
    def learn(self, states, actions, returns):
        with tf.GradientTape() as tape:
            action_probs = self.model(states)
            action_probs = tf.gather(action_probs, actions, batch_dims=1)
            action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
            log_probs = tf.math.log(action_probs)
            loss = -tf.reduce_mean(log_probs * returns)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def _compute_returns(self, rewards):
        returns = np.zeros_like(rewards)
        cumulative_return = 0
        for t in reversed(range(len(rewards))):
            cumulative_return = rewards[t] + self.gamma * cumulative_return
            returns[t] = cumulative_return
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns
    
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