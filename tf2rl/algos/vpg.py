import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.policies.tfp_gaussian_actor import GaussianActor

class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.conv_layers = [Conv2D(16, 3, strides=3, activation='relu'), Conv2D(64, 3, strides=2, activation='relu'), 
                            Conv2D(128, 3, strides=2, activation='relu'),
                            Conv2D(256, 3, strides=2, activation='relu'), GlobalAveragePooling2D()]
        self.connect_layers = [Dense(128, activation='relu'), Dense(32, activation='relu')]
        self.out_layer = Dense(1, name="V", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)
        #self.summary()

    def call(self, states):
        features = states
        for conv_layer in self.conv_layers:
            features = conv_layer(features) 

        for connect_layer in self.connect_layers:
            features = connect_layer(features)

        values = self.out_layer(features)

        return tf.squeeze(values, axis=1)


class VPG(OnPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            max_action=1.0,
            lr_actor=3e-4,
            lr_critic=3e-4,
            name="VPG",
            **kwargs):
        super().__init__(name=name, **kwargs)
        
        # set up actor and critic
        self.actor = GaussianActor(state_shape, action_dim, max_action, squash=True)
        self.critic = CriticV(state_shape)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor, clipnorm=5)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic, clipnorm=5)

        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

    def get_action(self, state, test=False):
        msg = "Input instance should be np.ndarray, not {}".format(type(state))
        assert isinstance(state, np.ndarray), msg

        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)

        action, logp, entropy = self._get_action_body(state, test)

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_and_val(self, state, test=False):
        is_single_input = state.ndim == self._state_ndim

        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)

        action, logp, v = self._get_action_logp_v_body(state, test)

        if is_single_input:
            v = v[0]
            action = action[0]

        return action.numpy(), logp.numpy(), v.numpy()

    @tf.function
    def _get_action_logp_v_body(self, state, test):
        action, logp, entropy = self.actor(state, test)
        v = self.critic(state)

        return action, logp, v

    @tf.function
    def _get_action_body(self, state, test):
        return self.actor(state, test)

    def train(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        actor_loss, logp_news = self._train_actor_body(states, actions, advantages, logp_olds)
        critic_loss = self._train_critic_body(states, returns)

        # Visualize results in TensorBoard
        tf.summary.scalar(name=self.policy_name+"/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max", data=np.max(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_min", data=np.min(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_mean", data=np.mean(logp_news))
        tf.summary.scalar(name=self.policy_name+"/adv_max", data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min", data=np.min(advantages))
        tf.summary.scalar(name=self.policy_name+"/kl", data=tf.reduce_mean(logp_olds - logp_news))
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=critic_loss)

        return actor_loss, critic_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                log_probs = self.actor.compute_log_probs(states, actions)
                weights = tf.stop_gradient(tf.squeeze(advantages))
                # + lambda * entropy
                actor_loss = tf.reduce_mean(-log_probs * weights)

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss, log_probs

    @tf.function
    def _train_critic_body(self, states, returns):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                current_V = self.critic(states)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(tf.square(td_errors))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
