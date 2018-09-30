# Deep Q-learning Agent
import collections
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
import random


class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 mem_len=5000,
                 learning_rate=0.01,
                 epsilon_decay=0.995,
                 layer_info=[]):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=mem_len)
        self.gamma = 0.8  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.train_count = 0
        self.score = 0
        if len(layer_info) < 1:
            self.layer_info = [
                self.state_size * self.state_size,
                self.state_size * self.action_size,
                self.state_size * self.action_size,
                self.action_size * self.action_size
            ]
        else:
            self.layer_info = list(layer_info)
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
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
        model.compile(
            loss='mse',
            optimizer=optimizers.Adam(lr=self.learning_rate, decay=0))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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
        old = self.epsilon
        self.epsilon *= r
        if self.epsilon > self.epsilon_decay:
            self.epsilon = self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        print("epsilon {:3f}->{:3f}".format(old, self.epsilon))

    def decrease_epsilon(self):
        return self.modify_epsilon(self.epsilon_decay)

    def increase_epsilon(self, max_epsilon=0.5):
        if self.epsilon > max_epsilon:
            return
        return self.modify_epsilon(2 - self.epsilon_decay)

    def act(self, state, isPlay=False):
        if ((not isPlay) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False
        act_values = self.model.predict(state)
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([i, act_values[0][i]])
        np.random.shuffle(act_result)
        return int(act_result[np.argmax(act_result, axis=0)[1]][0]), True

    def act_raw(self, state, isPlay=False):
        if ((not isPlay) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False, np.array([0])
        act_values = self.model.predict(state)
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([int(i), act_values[0][i]])
        np.random.shuffle(act_result)
        return int(act_result[np.argmax(act_result,
                                        axis=0)[1]][0]), True, np.array(
                                            act_values[0])

    def train(self, batch_size=32, mod_epsilon=True):
        batch_s = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_s)
        state_batch = np.zeros([batch_s, self.state_size])
        target_batch = np.zeros([batch_s, self.action_size])
        for i, (state, action, reward, next_state,
                done) in enumerate(minibatch):
            state_batch[i, :] = state
            target_batch[i, :] = self.model.predict(state)[0]
            if (not done) and ((state - next_state).any()):
                target_batch[i, int(action)] = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0])
            else:
                target_batch[i, int(action)] = reward
        self.model.fit(state_batch, target_batch, epochs=1, verbose=0)
        if (self.epsilon > self.epsilon_min) and mod_epsilon:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
        if self.train_count % 32 == 31:
            self.score = self.model.evaluate(
                state_batch, target_batch, verbose=0)

    def train_all(self, mod_epsilon=True):
        self.train(len(self.memory), mod_epsilon)
        self.memory.clear()
