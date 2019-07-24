# Reinforcement Agent
import collections

# @Hyper parameters
# =====================================================
EPSILON_INIT = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.01
GAMMA = 0.8
MEM_LEN = 1

# =====================================================


class AbstractAgent(object):
    ''' Interface of the agent '''

    def __init__(self,
                 state_size,
                 action_size,
                 mem_len=MEM_LEN,
                 gamma=GAMMA,
                 lr=LEARNING_RATE,
                 epsilon_decay=EPSILON_DECAY,
                 mdl_file=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=mem_len)
        self.gamma = gamma  # discount rate
        self.epsilon = EPSILON_INIT  # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        self.score = [0,0]
        self.train_count = 0
        self.mdl_file = mdl_file

    def remember(self, data):
        ''' storage the explore data '''
        self.memory.append(data)

    def modify_epsilon(self, r):
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
