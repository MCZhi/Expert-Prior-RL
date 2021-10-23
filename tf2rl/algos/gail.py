import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Concatenate

from tf2rl.algos.policy_base import IRLPolicy
from tf2rl.networks.spectral_norm_dense import SNDense


class Discriminator(tf.keras.Model):
    def __init__(self, state_shape, action_dim, output_activation="sigmoid", name="Discriminator"):
        super().__init__(name=name)
        
        self.conv_layers = [Conv2D(16, 3, strides=3, activation='relu'), Conv2D(64, 3, strides=2, activation='relu'), 
                            Conv2D(128, 3, strides=2, activation='relu'), Conv2D(256, 3, strides=2, activation='relu'), 
                            GlobalAveragePooling2D()]
        self.act_layers = [Dense(64, activation='relu')]
        self.connect_layers = [Dense(128, activation='relu'), Dense(32, activation='relu')]
        self.out_layer = Dense(1, name="reward", activation=output_activation)

        dummy_state = tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1, action_dim), dtype=np.float32))

        self(dummy_state, dummy_action)
        #self.summary()

    def call(self, state, action):
        features = state

        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        action = self.act_layers[0](action) 
        features_action = tf.concat([features, action], axis=1)

        for connect_layer in self.connect_layers:
            features_action = connect_layer(features_action)
        
        values = self.out_layer(features_action)

        return values

    def compute_reward(self, state, action):
        return tf.math.log(self(state, action) + 1e-8)


class GAIL(IRLPolicy):
    def __init__(
            self,
            state_shape,
            action_dim,
            lr=3e-4,
            enable_sn=False,
            n_training=1,
            name="GAIL",
            **kwargs):
        super().__init__(name=name, n_training=n_training, **kwargs)
        
        # set up discriminator
        self.disc = Discriminator(state_shape=state_shape, action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)

    def train(self, agent_states, agent_acts, expert_states, expert_acts, **kwargs):
        loss, accuracy, js_divergence, inference_reward, real_logits, fake_logits = self._train_body(
            agent_states, agent_acts, expert_states, expert_acts)

        tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
        tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
        tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)
        tf.summary.scalar(name=self.policy_name+"/InferenceReward", data=inference_reward)
        tf.summary.scalar(name=self.policy_name+"/RealLogits", data=real_logits)
        tf.summary.scalar(name=self.policy_name+"/FakeLogits", data=fake_logits)

    def _compute_js_divergence(self, fake_logits, real_logits):
        m = (fake_logits + real_logits) / 2.0
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.0)

    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts):
        epsilon = 1e-8

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                real_logits = self.disc(expert_states, expert_acts)
                fake_logits = self.disc(agent_states, agent_acts)

                loss = -(tf.reduce_mean(tf.math.log(real_logits + epsilon)) +
                         tf.reduce_mean(tf.math.log(1.0 - fake_logits + epsilon)))

            grads = tape.gradient(loss, self.disc.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.disc.trainable_variables))

        accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2.0 +
                    tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.0)

        js_divergence = self._compute_js_divergence(fake_logits, real_logits)
        inference_reward = self.disc.compute_reward(agent_states, agent_acts)

        return loss, accuracy, js_divergence, tf.reduce_mean(inference_reward), tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

    def inference(self, states, actions, next_states):
        states = np.expand_dims(states, axis=0)
        actions = np.expand_dims(actions, axis=0)

        return self._inference_body(states, actions)

    @tf.function
    def _inference_body(self, states, actions):
        with tf.device(self.device):
            return self.disc.compute_reward(states, actions)

    @staticmethod
    def get_argument(parser=None):
        parser = IRLPolicy.get_argument(parser)
        parser.add_argument('--enable-sn', action='store_true')
        
        return parser
