# Deep Q-learning Agent
import tensorflow as tf
import numpy as np
import random
import os
from agents.abstract_agent import AbstractAgent

# @Hyper parameters
# =====================================================
EPSILON_DECAY = 0.995
GAMMA = 0.8
LEARNING_RATE = 0.01
MEM_LEN = 2000
LAYER_INFO = [64, 256, 128, 32]
EVALUATE_EPISODE = 8
BATCH_SIZE = 64
PARAM_FILE = "{}.para"

# =====================================================


class Agent(AbstractAgent):
    def __init__(self,
                 state_size,
                 action_size,
                 mem_len=MEM_LEN,
                 gamma=GAMMA,
                 learning_rate=LEARNING_RATE,
                 epsilon_decay=EPSILON_DECAY,
                 mdl_file=None):
        ''' Deep q-learning agnet '''

        super(Agent, self).__init__(state_size=state_size,
                                    action_size=action_size,
                                    mem_len=mem_len,
                                    gamma=gamma,
                                    epsilon_decay=epsilon_decay,
                                    mdl_file=mdl_file)

        if self.mdl_file:
            if os.path.exists(mdl_file):
                self.model = tf.keras.models.load_model(mdl_file)
            else:
                self.model = self._build_model()
            if os.path.exists(PARAM_FILE.format(mdl_file)):
                with open(PARAM_FILE.format(mdl_file)) as f:
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
        # model.add(tf.keras.layers.Dropout(0.2))
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

    def train(self, mod_epsilon=True):
        batch_s = min(BATCH_SIZE, len(self.memory))
        minibatch = random.sample(self.memory, batch_s)
        state_batch = np.zeros([batch_s, self.state_size])
        target_batch = np.zeros([batch_s, self.action_size])
        for i, (state, action, reward, next_state,
                done) in enumerate(minibatch):
            state_batch[i, :] = state
            target_batch[i, :] = self.model.predict(
                np.reshape(state, [1, self.state_size]))[0]
            if (not done) and ((state - next_state).any()):
                target_batch[i, action] = reward + self.gamma * np.amax(
                    self.model.predict(
                        np.reshape(next_state, [1, self.state_size]))[0])
            else:
                target_batch[i, action] = reward
        self.model.fit(state_batch, target_batch, epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min) and mod_epsilon:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
        if self.train_count % EVALUATE_EPISODE == EVALUATE_EPISODE - 1:
            self.score = self.model.evaluate(state_batch,
                                             target_batch,
                                             verbose=0)

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
