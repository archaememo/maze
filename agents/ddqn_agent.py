# Deep Q-learning Agent
import collections
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K

import numpy as np
import random


class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 mem_len=5000,
                 learning_rate=0.01,
                 epsilon_decay=0.995,
                 layer_info=[],
                 is_ddqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=mem_len)
        self.gamma = 0.8  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        if len(layer_info) < 1:
            self.layer_info = [
                self.state_size * self.state_size,
                self.state_size * self.action_size,
                self.state_size * self.action_size,
                self.action_size * self.action_size
            ]
        else:
            self.layer_info = list(layer_info)
        self.is_ddqn = is_ddqn
        self.train_count = 0
        self._build_model()

    # Actor model
    def _build_model(self):
        self.eval_model = self._create_mode()
        self.target_model = self._create_mode()
        adam = Adam(lr=self.learning_rate)
        self.eval_model.compile(loss="mse", optimizer=adam)
        self.eval_model._make_predict_function()
        self.target_model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.session = K.get_session()
        self.score = 0

    def _create_mode(self):
        model = Sequential()
        model.add(
            Dense(
                self.layer_info[0],
                input_dim=self.state_size,
                activation='relu'))
        # add middle hidden layer
        for i in range(1, len(self.layer_info)):
            model.add(Dense(self.layer_info[i], activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        return model

    def remember(self, data):
        self.memory.append(data)

    def forget(self, num):
        for _ in range(min(num, len(self.memory))):
            self.memory.pop()

    def enforce_memory(self, back_start=16, num=16, enforce_times=4):
        len_m = len(self.memory)
        if (back_start < 1) or (back_start > len_m) or (
                enforce_times <= 0) or (num > back_start):
            return
        cache_memory = []
        for n in range(len_m - back_start, len_m):
            cache_memory.append(self.memory[n])
        for n in range(num):
            for _ in range(enforce_times):
                self.memory.append(cache_memory[n])

    def reset_epsilon(self, r=0.9):
        self.epsilon = r

    def modify_epsilon(self, r=1.01):
        # old = self.epsilon
        self.epsilon *= r
        if self.epsilon > self.epsilon_decay:
            self.epsilon = self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        # print("epsilon {:3f}->{:3f}".format(old, self.epsilon))

    def decrease_epsilon(self):
        return self.modify_epsilon(self.epsilon_decay)

    def increase_epsilon(self):
        return self.modify_epsilon(2 - self.epsilon_decay)

    def act(self, state, isPlay=False):
        if ((not isPlay) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False
        with self.session.as_default():
            with self.graph.as_default():
                act_values = self.eval_model.predict(state)
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([i, act_values[0][i]])
        np.random.shuffle(act_result)
        return int(act_result[np.argmax(act_result, axis=0)[1]][0]), True

    def act_raw(self, state, isPlay=False):
        if ((not isPlay) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False, np.array([0])
        with self.session.as_default():
            with self.graph.as_default():
                act_values = self.eval_model.predict(state)
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([int(i), act_values[0][i]])
        np.random.shuffle(act_result)
        return int(act_result[np.argmax(
            act_result, axis=0)[1]][0]), True, np.array(act_values[0])

    def train(self, batch_size=32, mod_epsilon=True):
        batch_s = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_s)
        random.shuffle(minibatch)
        state_batch = np.zeros([batch_s, self.state_size])
        target_batch = np.zeros([batch_s, self.action_size])
        with self.session.as_default():
            with self.graph.as_default():
                for i, (state, action, reward, next_state,
                        done) in enumerate(minibatch):
                    state_batch[i, :] = state
                    target_batch[i, :] = self.eval_model.predict(state)[0]
                    if (not done) and ((state - next_state).any()):
                        next_q = self.target_model.predict(next_state)[0]
                        if self.is_ddqn:
                            next_a = self.act(next_state, isPlay=True)
                            next_max = next_q[next_a]
                        else:
                            next_max = np.amax(next_q)
                        target_batch[i, int(
                            action)] = reward + self.gamma * next_max
                    else:
                        target_batch[i, int(action)] = reward
                self.eval_model.fit(
                    state_batch, target_batch, epochs=1, verbose=0)
                if (self.epsilon > self.epsilon_min) and mod_epsilon:
                    self.epsilon *= self.epsilon_decay
                self.train_count += 1
                if self.train_count % 32 == 31:
                    self._update_target()
                    self.score = self.eval_model.evaluate(
                        state_batch, target_batch, verbose=0)

    def train_all(self, mod_epsilon=True):
        self.train(len(self.memory), mod_epsilon)
        self.memory.clear()

    def _update_target(self):
        with self.session.as_default():
            with self.graph.as_default():
                eval_weights = self.eval_model.get_weights()
                target_weights = self.target_model.get_weights()
                for i in range(len(target_weights)):
                    target_weights[i] = eval_weights[i]
                self.target_model.set_weights(target_weights)
