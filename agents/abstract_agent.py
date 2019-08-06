# Reinforcement Agent
import collections
from config import (EPSILON_INIT, EPSILON_MIN, EPSILON_DECAY, LEARNING_RATE,
                    GAMMA, MEM_LEN)


class AbstractAgent(object):
    ''' Interface of the agent '''

    def __init__(self,
                 state_size,
                 action_size,
                 mdl_file=None,
                 gamma=GAMMA,
                 learning_rate=LEARNING_RATE):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=MEM_LEN)
        self.gamma = gamma  # discount rate
        self.epsilon = EPSILON_INIT  # exploration rate

        self.learning_rate = learning_rate
        self.score = [0, 0]
        self.train_count = 0
        self.mdl_file = mdl_file

    def remember(self, data):
        ''' storage the explore data '''
        self.memory.append(data)

    def modify_epsilon(self, epsilon):
        self.epsilon = epsilon
        if self.epsilon > EPSILON_DECAY:
            self.epsilon = EPSILON_DECAY
        if self.epsilon < EPSILON_MIN:
            self.epsilon = EPSILON_MIN

    def modify_epsilon_by_rate(self, rate):
        self.modify_epsilon(self.epsilon * rate)

    def decrease_epsilon(self):
        return self.modify_epsilon_by_rate(EPSILON_DECAY)

    def increase_epsilon(self, rate=None):
        return self.modify_epsilon_by_rate(2 - EPSILON_DECAY)

    def act_raw(self, state, predict_only=False):
        ''' choose the action according to state, and return extra info '''
        action = 0
        is_predict = False
        info = 0
        return action, is_predict, info

    def act(self, state, predict_only=False):
        result, is_predict, _ = self.act_raw(state, predict_only=False)
        return result, is_predict

    def train(self):
        ''' training the agent '''
        self.decrease_epsilon()
