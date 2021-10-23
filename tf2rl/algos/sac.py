import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Concatenate

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.policies.tfp_gaussian_actor import GaussianActor


class CriticV(tf.keras.Model):
    def __init__(self, state_shape, name='vf'):
        super().__init__(name=name)

        self.conv_layers = [Conv2D(16, 3, strides=3, activation='relu'), Conv2D(64, 3, strides=2, activation='relu'), 
                            Conv2D(128, 3, strides=2, activation='relu'), Conv2D(256, 3, strides=2, activation='relu'), 
                            GlobalAveragePooling2D()]
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


class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, name='qf'):
        super().__init__(name=name)

        self.conv_layers = [Conv2D(16, 3, strides=3, activation='relu'), Conv2D(64, 3, strides=2, activation='relu'), 
                            Conv2D(128, 3, strides=2, activation='relu'), Conv2D(256, 3, strides=2, activation='relu'), 
                            GlobalAveragePooling2D()]
        self.act_layers = [Dense(64, activation='relu')]
        self.connect_layers = [Dense(128, activation='relu'), Dense(32, activation='relu')]
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)
        #self.summary()

    def call(self, states, actions):
        features = states

        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        action = self.act_layers[0](actions) 
        features_action = tf.concat([features, action], axis=1)

        for connect_layer in self.connect_layers:
            features_action = connect_layer(features_action)

        values = self.out_layer(features_action)

        return tf.squeeze(values, axis=1)


class SAC(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="SAC",
            max_action=1.0,
            lr=3e-4,
            tau=5e-3,
            alpha=0.2,
            auto_alpha=False,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):
        super().__init__(
            name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        self._setup_actor(state_shape, action_dim, lr, max_action)
        self._setup_critic_v(state_shape, lr)
        self._setup_critic_q(state_shape, action_dim, lr)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32)
            self.alpha = tfp.util.DeferredTensor(pretransformed_input=self.log_alpha, transform_fn=tf.exp, dtype=tf.float32)
            self.target_alpha = -action_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        else:
            self.alpha = alpha

        self.state_ndim = len(state_shape)

    def _setup_actor(self, state_shape, action_dim, lr, max_action=1.):
        self.actor = GaussianActor(state_shape, action_dim, max_action, squash=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_q(self, state_shape, action_dim, lr):
        self.qf1 = CriticQ(state_shape, action_dim, name="qf1")
        self.qf2 = CriticQ(state_shape, action_dim, name="qf2")
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_v(self, state_shape, lr):
        self.vf = CriticV(state_shape)
        self.vf_target = CriticV(state_shape)
        update_target_variables(self.vf_target.weights, self.vf.weights, tau=1.0)
        self.vf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def get_action(self, state, test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state), test)

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state, test):
        actions, log_pis, entropy = self.actor(state, test)

        return actions

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, actor_loss, vf_loss, qf_loss, q_value, logp_min, logp_max, logp_mean, entropy_mean = self._train_body(
            states, actions, next_states, rewards, dones, weights)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_V_loss", data=vf_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_Q_loss", data=qf_loss)
        tf.summary.scalar(name=self.policy_name + "/Q_value", data=q_value)
        tf.summary.scalar(name=self.policy_name + "/logp_min", data=logp_min)
        tf.summary.scalar(name=self.policy_name + "/logp_max", data=logp_max)
        tf.summary.scalar(name=self.policy_name + "/logp_mean", data=logp_mean)
        tf.summary.scalar(name=self.policy_name + "/entropy_mean", data=entropy_mean)
        if self.auto_alpha:
            tf.summary.scalar(name=self.policy_name + "/logp_mean+target", data=logp_mean + self.target_alpha)
        tf.summary.scalar(name=self.policy_name + "/alpha", data=self.alpha)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2
            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)

            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Compute loss of critic Q
                current_q1 = self.qf1(states, actions)
                current_q2 = self.qf2(states, actions)
                next_v_target = self.vf_target(next_states)

                target_q = tf.stop_gradient(rewards + not_dones * self.discount * next_v_target)

                td_loss_q1 = tf.reduce_mean((target_q - current_q1) ** 2)
                td_loss_q2 = tf.reduce_mean((target_q - current_q2) ** 2) 

                # Compute loss of critic V
                current_v = self.vf(states)

                # Resample actions to update V
                sample_actions, logp, entropy = self.actor(states)  
                current_q1 = self.qf1(states, sample_actions)
                current_q2 = self.qf2(states, sample_actions)
                current_min_q = tf.minimum(current_q1, current_q2)

                target_v = tf.stop_gradient(current_min_q - self.alpha * logp)

                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(td_errors ** 2)

                # Compute loss of policy
                policy_loss = tf.reduce_mean(self.alpha * logp - current_min_q)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean((self.alpha * tf.stop_gradient(logp + self.target_alpha)))

            # Critic Q1 loss
            q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))

            # Critic Q2 loss
            q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

            # Critic V loss
            vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
            self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))
            # Update Target V
            update_target_variables(self.vf_target.weights, self.vf.weights, self.tau)

            # Actor loss
            actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Alpha loss
            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
          
            del tape

        return td_errors, policy_loss, td_loss_v, td_loss_q1, tf.reduce_mean(current_min_q), tf.reduce_min(logp), \
                tf.reduce_max(logp), tf.reduce_mean(logp), tf.reduce_mean(entropy)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)

        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)

        return td_errors.numpy()

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            # Compute TD errors for Q-value func
            current_q1 = self.qf1(states, actions)
            vf_next_target = self.vf_target(next_states)

            target_q = tf.stop_gradient(rewards + not_dones * self.discount * vf_next_target)

            td_errors_q1 = target_q - current_q1

        return td_errors_q1

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--auto-alpha', action="store_true")

        return parser
