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

    def act(self, state, test=False):
        probs = self._act(state).numpy()  # Convert to NumPy
        if self.num_envs == 1 or test:
            action = np.random.choice(probs.shape[-1], p=probs[0])
            return action
        else:
            actions = [np.random.choice(p.shape[-1], p=p) for p in probs]
            return np.array(actions)
    
    @tf.function
    def learn(self, states, actions, rewards, next_states, dones, I):
        dones = tf.cast(dones, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        states = tf.cast(states, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        actions = tf.cast(actions, tf.int32)
        with tf.GradientTape(persistent=True) as tape:
            probs = self.actor(states)  # (batch_size, num_actions)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.output_shape[0]), axis=1)
            action_probs = tf.clip_by_value(action_probs, 1e-10, 1.0)
            log_probs = tf.math.log(action_probs)
            critic_current = self.critic(states)
            target_critic = rewards + (1.0 - dones) * self.gamma * self.critic(next_states)
            target_critic = tf.stop_gradient(target_critic)
            advantages = target_critic - critic_current
            actor_loss = -tf.reduce_mean(log_probs * I * advantages)
            actor_loss -= 0.01 * tf.reduce_mean(probs * tf.math.log(probs + 1e-10)) 
            critic_loss = tf.reduce_mean(tf.square(advantages))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads = [tf.clip_by_value(grad, -5.0, 5.0) for grad in actor_grads]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads = [tf.clip_by_value(grad, -5.0, 5.0) for grad in critic_grads]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return actor_loss + critic_loss

    
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