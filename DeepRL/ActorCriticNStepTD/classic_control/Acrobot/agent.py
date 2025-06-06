import tensorflow as tf
import numpy as np
import os

from model import *
from config import *
from buffer import *

class ActorCriticAgent:
    def __init__(self,input_shape=ENV_INPUT_SHAPE,output_shape=ENV_OUTPUT_SHAPE,num_envs=NUM_ENVS,learning_rate=ALPHA,gamma=GAMMA):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer = TrajectoryBuffer(num_envs=num_envs)
        self.actor = ActorModel(input_shape=self.input_shape,output_shape=self.output_shape)
        self.critic = CriticModel(input_shape=self.input_shape, output_shape=(1,))
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def _act(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        probs = self.actor(state)
        return probs
    
    @tf.function
    def _critic_value(self, state):
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        value = self.critic(state)
        return value

    def act(self, state, test=False):
        probs = self._act(state).numpy()  # Convert to NumPy
        if self.num_envs == 1 or test:
            action = np.random.choice(probs.shape[-1], p=probs[0])
            return action
        else:
            actions = [np.random.choice(p.shape[-1], p=p) for p in probs]
            return np.array(actions)
        
    def _compute_returns(self, rewards, final_state, done, gamma=0.99):
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0.0

        if not done:
            final_state = np.expand_dims(final_state, axis=0)  # batch dim
            V_final = self._critic_value(final_state).numpy()[0]
            G = V_final

        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns[t] = G

        return returns
    
    
    @tf.function
    def learn(self, states, actions, returns):
        # Convert to tensors
        states = tf.cast(states, dtype=tf.float32)         # [n, obs_dim]
        actions = tf.cast(actions, dtype=tf.int32)         # [n]
        returns = tf.cast(returns, dtype=tf.float32)       # [n]

        with tf.GradientTape(persistent=True) as tape:
            probs = self.actor(states)                                   # [n, num_actions]
            log_probs = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0)) # [n, num_actions]
            one_hot = tf.one_hot(actions, self.output_shape[0])          # [n, num_actions]
            selected_log_probs = tf.reduce_sum(log_probs * one_hot, axis=1)  # [n]
            values = tf.squeeze(self.critic(states), axis=1)             # [n]
            advantages = returns - values                                # [n]
            actor_loss = -tf.reduce_mean(selected_log_probs * tf.stop_gradient(advantages))
            # entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1))
            total_actor_loss = actor_loss #+ 0.01 * entropy
            critic_loss = tf.reduce_mean(tf.square(advantages))

        actor_grads = tape.gradient(total_actor_loss, self.actor.trainable_variables)
        actor_grads = [tf.clip_by_value(g, -5.0, 5.0) for g in actor_grads]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads = [tf.clip_by_value(g, -5.0, 5.0) for g in critic_grads]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return total_actor_loss + critic_loss

    
    def save_model(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        actor_path = 'models/actor_weights.weights.h5'
        critic_path = 'models/critic_weights.weights.h5'
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Model saved to {actor_path} and {critic_path}")
        
    def load_model(self):
        actor_path = 'models/actor_weights.weights.h5'
        critic_path = 'models/critic_weights.weights.h5'
        if not os.path.exists(actor_path):
            self.save_model()
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        print(f"Model loaded from {actor_path} and {critic_path}")