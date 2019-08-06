# Deep Q-learning Agent
import tensorflow as tf
import numpy as np
import random
import os
from agents.abstract_agent import AbstractAgent
from config import (PARAM_FILE, LAYER_INFO)

# @Hyper parameters
# =====================================================
BATCH_SIZE = 64
# =====================================================

# tf.enable_eager_execution()


class Agent(AbstractAgent):
    def __init__(self, **kwargs):
        ''' Deep q-learning agnet '''

        super(Agent, self).__init__(**kwargs)

        if self.mdl_file:
            if os.path.exists(self.mdl_file):
                self.model = tf.keras.models.load_model(self.mdl_file)
            else:
                self.model = self._build_model()
            if os.path.exists(PARAM_FILE.format(self.mdl_file)):
                with open(PARAM_FILE.format(self.mdl_file)) as f:
                    self.epsilon = float(f.readline())
            self.best_accuracy = 0
        else:
            self.model = self._build_model()

    def _build_model(self, layer_info=LAYER_INFO):
        # Neural Net for Deep-Q learning Model

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(layer_info[0],
                                  activation='relu',
                                  input_dim=self.state_size))
        for i in range(1, len(layer_info)):
            model.add(tf.keras.layers.Dense(layer_info[i], activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.categorical_accuracy])
        return model

    def act_raw(self, state, predict_only=False):
        if ((not predict_only) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False, np.array([0])
        act_values = self.model.predict(np.reshape(state,
                                                   [1, self.state_size]))
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([int(i), act_values[0][i]])
        np.random.shuffle(act_result)
        return int(act_result[np.argmax(
            act_result, axis=0)[1]][0]), True, np.array(act_values[0])

    def generate_samples(self, size):
        batch_s = min(size, len(self.memory))
        minibatch = random.sample(self.memory, batch_s)
        state_batch = np.zeros([batch_s, self.state_size])
        target_batch = np.zeros([batch_s, self.action_size])
        for i, (state, action, reward, next_state,
                done) in enumerate(minibatch):
            state_batch[i, :] = state
            target_batch[i, :] = self.model.predict(
                np.reshape(state, [1, self.state_size]))[0]
            if (not done) and ((state - next_state).any()):
                target_batch[i, action] = max(
                    0, reward + max(
                        0,
                        self.gamma * np.amax(
                            self.model.predict(
                                np.reshape(next_state,
                                           [1, self.state_size]))[0])))
            else:
                target_batch[i, action] = reward
        return state_batch, target_batch

    def train(self):
        state_batch, target_batch = self.generate_samples(BATCH_SIZE)
        self.model.fit(state_batch, target_batch, epochs=1, verbose=0)

    def evaluate(self):
        state_batch, target_batch = self.generate_samples(BATCH_SIZE)
        self.score = self.model.evaluate(state_batch, target_batch, verbose=0)
        return self.score

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_model(self, mdl_file=None):
        if not mdl_file:
            mdl_file = self.mdl_file
        if mdl_file:
            self.model.save(mdl_file)
            with open(PARAM_FILE.format(mdl_file), "w+") as f:
                f.write(str(self.epsilon))
                f.write("\n")
        else:
            pass
