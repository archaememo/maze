# Q-learning Agent
import numpy as np
import random


class QLAgent:
    def __init__(self, state_size, action_size, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.q_map = np.zeros([state_size, action_size])
        self.gamma = 0.8  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay

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

    def increase_epsilon(self):
        return self.modify_epsilon(2 - self.epsilon_decay)

    def act(self, state, isPlay=False):
        if ((not isPlay) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False
        act_values = self.q_map[state][:]
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([int(i), act_values[i]])
        np.random.shuffle(act_result)
        return act_result[np.argmax(act_result, axis=0)[1]][0], True

    def act_raw(self, state, isPlay=False):
        if ((not isPlay) and (np.random.rand() <= self.epsilon)):
            return random.randrange(self.action_size), False, [0]
        act_values = self.q_map[state][:]
        act_result = np.zeros([self.action_size, 2])
        for i in range(self.action_size):
            act_result[i][:] = np.array([int(i), act_values[i]])
        np.random.shuffle(act_result)
        return act_result[np.argmax(act_result,
                                    axis=0)[1]][0], True, act_values

    def remember(self, state, action, reward, next_state, done):
        if (not done) and state != next_state:
            self.q_map[state][action] = reward + self.gamma * np.amax(
                self.q_map[next_state][:])
        else:
            self.q_map[state][action] = reward
