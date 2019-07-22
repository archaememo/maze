# import numpy as np
import time
# import threading
# import copy
# import collections
# import sys
# from env.maze import Maze
# from agents.if_agent import AgentInterface

# @Hyper parameters
# =====================================================
DISPLAY_DELAY = 0.01
GRID_SIZE = 8
BLOCK_NUM = 8
EXPLORE_NUM = 1
MAX_EPISODE = 100000
PIXEL_UNIT = 40
INC_EPISODE_THRESHOLD = 16
INC_EPISODE_RATE = 1.2
VERIFY_EPISODE = 32
FEATURE_TYPE = 3
VERIFY_THRESHOLD = 3
VERIFY_TIME = 2


# =====================================================
class AbstractPlayer(object):
    def __init__(self,
                 AgentClass,
                 maze,
                 delay=DISPLAY_DELAY,
                 quiet=False,
                 explorer_num=EXPLORE_NUM,
                 confirm_stop=True,
                 factor=1):
        ''' init function of class MazePlayer'''
        # initilization
        self.AgentClass = AgentClass
        self.maze = maze
        self.n_features = maze.observation_space.shape[0]
        self.n_actions = maze.action_space.n
        self.delay_default = delay
        self.quiet = quiet
        self.n_explorers = explorer_num
        self.max_step = int(maze.height * maze.width * 4)
        self.episode = MAX_EPISODE
        self.learn_batch = int(maze.height + maze.width)
        self.pixel_unit = PIXEL_UNIT
        self.start_t = time.time()
        self.confirm_stop = confirm_stop
        self.wait_verify = False
        self.factor = factor
        self._init_agent()

    def _init_agent(self):
        self.agent = self.AgentClass(self.n_features,
                                     self.n_actions,
                                     factor=self.factor)

    def explore_process(self, env):
        done = False
        state = env.reset()
        cache = []
        for _ in range(self.max_step):
            action, _ = self.agent.act(state)
            next_state, reward, done = env.step(action, quiet=self.quiet)
            try:
                cache.index((state, action, reward, next_state, done))
                continue
            except ValueError:
                cache.append((state, action, reward, next_state, done))

            if done and (reward > 0):
                # enforce success memory
                for _ in range(8):
                    for r in cache[-8:]:
                        self.agent.remember(r)
                break
            state = next_state
        for r in cache:
            try:
                self.agent.memory.index(r)
            except ValueError:
                self.agent.remember(r)
        return done and (reward > 0)

    def train_process(self, mod_epsilon=True):
        self.agent.train(mod_epsilon)

    def run(self):
        self.done_t = 0
        self.lost_t = 0
        self.verify_t = 0
        self.delay_t = self.delay_default
        self.last_steps = 0
        self.cur_episode = 0

        # if not self.quiet:
        #     verify_exp = self.verify_exp
        # else:
        #     verify_exp = self.verify_exp_quiet

        for self.cur_episode in range(self.episode):
            if not self.wait_verify:
                mod_epsilon = self.explore_process(self.maze)
                self.train_process(mod_epsilon)
            if (self.cur_episode % VERIFY_EPISODE == VERIFY_EPISODE -
                    1) or self.wait_verify:
                reward, done, steps = self.verify_exp(self.delay_t)
                if done and (reward > 0):
                    if self.verify_suc_proc():
                        break
                else:
                    self.verify_fail_proc(steps)
                if not self.quiet:
                    self.statistic()
        self.statistic()
        self.verify_exp(0.5)
        self.exit_process()

    def exit_process(self):
        if self.confirm_stop:
            if not self.quiet:
                while True:
                    time.sleep(0.5)
                    self.maze.render()
            else:
                input()
        return self.cur_episode

    def verify_suc_proc(self):
        if self.wait_verify:
            self.verify_t += 1
            if self.verify_t > VERIFY_TIME - 1:
                print("[{}] success to train mode to find goal...".format(
                    self.cur_episode))
                return True
        else:
            self.done_t += 1
            if self.done_t >= VERIFY_THRESHOLD:
                self.wait_verify = True
                print("[{}] Enter into verification status...".format(
                    self.cur_episode))
                self.delay_t = 0.5
                self.done_t = 0
        self.agent.modify_epsilon(0.8)
        self.lost_t = 0
        return False

    def verify_fail_proc(self, steps):
        if steps <= self.last_steps:
            # step less than or equal to before, not good
            self.lost_t += 1
            if self.lost_t > INC_EPISODE_THRESHOLD:
                # increase epsilon rate if stay in bad status
                self.agent.modify_epsilon(INC_EPISODE_RATE)
                self.lost_t = 0
        else:
            # step farther
            self.lost_t = max(self.lost_t - 1, 0)
        # Record the latest steps
        self.last_steps = steps
        self.done_t = max(self.done_t - 1, 0)
        if self.wait_verify:
            self.wait_verify = False
            print("[{}] Out of verification status...".format(
                self.cur_episode))
            self.verify_t = 0
            self.delay_t = self.delay_default

    def verify_exp(self, deplay_time=0.01):
        i = 0
        done = False
        state = self.maze.reset()
        self.maze.render()
        track = [list(state)]
        while not done:
            i = i + 1
            action, _ = self.agent.act(state, True)
            state, reward, done = self.maze.step(action, quiet=False)
            if list(state) in track:
                break
            else:
                track.append(list(state))
            self.maze.render()
            time.sleep(deplay_time)
            if i > self.max_step:
                break
        return reward, done, i

    # def verify_exp_quiet(self, deplay_time=0.1):
    #     i = 0
    #     done = False
    #     state = self.maze.reset()
    #     track = [list(state)]
    #     while not done:
    #         i = i + 1
    #         action, _ = self.agent.act(state, True)
    #         state, reward, done = self.maze.step(action,
    #                                              print_track=print_track)
    #         # state, _ = self.maze.get_features()
    #         if list(state) in track:
    #             break
    #         else:
    #             track.append(list(state))
    #         if i > self.max_step:
    #             break
    #         # have to wait for a while to avoid keras crash
    #         # time.sleep(deplay_time)
    #     return reward, done, i

    def statistic(self):
        print("[{}] time:{:.2f}, steps:{}, epsilon:{:.3f}, "
              "score:{:.3f}, train_c:{}".format(self.cur_episode,
                                                time.time() - self.start_t,
                                                self.last_steps,
                                                self.agent.epsilon,
                                                self.agent.score[0],
                                                self.agent.train_count))
