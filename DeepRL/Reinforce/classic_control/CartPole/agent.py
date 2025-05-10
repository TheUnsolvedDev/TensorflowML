import tensorflow as tf
import numpy as np
import os

from model import *
from config import *
from buffer import *

class ReinforceAgent:
    def __init__(self,input_shape=ENV_INPUT_SHAPE,output_shape=ENV_OUTPUT_SHAPE,num_envs=NUM_ENVS,learning_rate=ALPHA,gamma=GAMMA):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.model = PolicyModel(input_shape=self.input_shape,output_shape=self.output_shape)
        self.baseline_model = BaselineModel(input_shape=self.input_shape, output_shape=(1,))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
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
        with tf.GradientTape(persistent=True) as tape:
            logits = self.model(states)  # (batch_size, num_actions)
            action_probs = tf.reduce_sum(logits * tf.one_hot(actions, self.output_shape[0]), axis=1)
            action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
            log_probs = tf.math.log(action_probs)
            baseline_values = tf.squeeze(self.baseline_model(states), axis=-1)  # (batch_size,)
            advantages = returns - baseline_values
            policy_loss = -tf.reduce_mean(log_probs * advantages)
            baseline_loss = tf.reduce_mean(tf.square(advantages))

        policy_grads = tape.gradient(policy_loss, self.model.trainable_variables)
        policy_grads = [tf.clip_by_value(grad, -5.0, 5.0) for grad in policy_grads]
        self.optimizer.apply_gradients(zip(policy_grads, self.model.trainable_variables))

        baseline_grads = tape.gradient(baseline_loss, self.baseline_model.trainable_variables)
        baseline_grads = [tf.clip_by_value(grad, -5.0, 5.0) for grad in baseline_grads]
        self.baseline_optimizer.apply_gradients(zip(baseline_grads, self.baseline_model.trainable_variables))
        return policy_loss + baseline_loss

    
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