import silence_tensorflow.auto
import gymnax
import tensorflow as tf
import jax
import functools

key = jax.random.PRNGKey(0)


class Env:
    def __init__(self, num_envs=8, ENV='CartPole-v1') -> None:
        self.env, self.env_params = gymnax.make(ENV)
        self.vmap_keys = jax.random.split(key, num_envs)
        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self):
        obs, state = self.vmap_reset(self.vmap_keys, self.env_params)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        next_obs, next_state, reward, done, _ = self.vmap_step(
            self.vmap_keys, state, action, self.env_params)
        return next_obs, next_state, reward, done, _


class PolicyGradient:
    def __init__(self,):
        self.env = Env(1)
        self.total_steps = int(2*1e+6)
        self.max_steps_per_episode = int(1e+3)

    def policy_loss(actions, discounted_rewards, logits):
        actions_one_hot = tf.one_hot(actions, depth=logits.shape[1])
        advantages = discounted_rewards - tf.reduce_mean(discounted_rewards)
        policy_loss = -tf.reduce_sum(tf.math.log(tf.reduce_sum(
            actions_one_hot * tf.exp(logits), axis=1)) * advantages)
        return policy_loss

    def train(self,):
        pass

    def test(self,):
        pass


if __name__ == "__main__":
    env = Env(1)
    print(env.reset())
