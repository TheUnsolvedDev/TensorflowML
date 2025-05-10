import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

from model import *
from config import *
from buffer import *

class ReinforceAgent:
    def __init__(self,input_shape=ENV_INPUT_SHAPE,num_envs=NUM_ENVS,learning_rate=ALPHA,gamma=GAMMA):
        self.input_shape = input_shape
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.model = PolicyModel(input_shape=self.input_shape)
        self.baseline_model = BaselineModel(input_shape=self.input_shape, output_shape=(1,))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.trajectory_buffer = TrajectoryBuffer(num_envs=self.num_envs)

    @tf.function
    def _act(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        mus, stds = self.model(state)
        return mus, stds
    
    def act(self, state, test=False):
        mus, stds = self._act(state)  # Convert to NumPy
        action_dist = tfp.distributions.Normal(loc=mus, scale=stds)
        action = action_dist.sample()
        action = np.clip(action.numpy(), ACTION_MIN, ACTION_MAX)
        if self.num_envs == 1 or test:
            return action[0]
        else:
            return action
    
    @tf.function
    def learn(self, states, actions, returns):
        with tf.GradientTape(persistent=True) as tape:
            mus,stds = self.model(states)
            distributions = tfp.distributions.Normal(loc=mus, scale=stds)
            action_probs = distributions.prob(actions)
            action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
            log_probs = tf.math.log(action_probs)
            baseline_values = tf.squeeze(self.baseline_model(states), axis=-1)  # (batch_size,)
            advantages = returns - baseline_values
            loss = -tf.reduce_mean(log_probs * returns)
            entropy = distributions.entropy()
            loss -= 0.01 * tf.reduce_mean(entropy)
            baseline_loss = tf.reduce_mean(tf.square(advantages))
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        baseline_grads = tape.gradient(baseline_loss, self.baseline_model.trainable_variables)
        baseline_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in baseline_grads]
        self.baseline_optimizer.apply_gradients(zip(baseline_grads, self.baseline_model.trainable_variables))
        return loss + baseline_loss
    
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
        
if __name__ == '__main__':
    state = np.random.rand(1, ENV_INPUT_SHAPE[0])
    agent = PolicyGradientAgent()