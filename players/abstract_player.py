import time
import random
import numpy as np
from config import (
    MAX_EPISODE,
    DISPLAY_DELAY,
    ACCURACY_THRESHOLD,
    VERIFY_EPISODE,
    VERIFY_TIME,
    VERIFY_THRESHOLD,
)


class AbstractPlayer(object):
    def __init__(self,
                 AgentClass,
                 maze,
                 quiet=False,
                 confirm_stop=True,
                 mdl_file=None):
        ''' init function of class MazePlayer'''
        # initilization
        self.maze = maze
        self.quiet = quiet
        self.max_step = int(maze.height * maze.width)
        self.episode = MAX_EPISODE
        self.learn_batch = int(maze.height + maze.width)
        self.start_t = time.time()
        self.confirm_stop = confirm_stop
        self.wait_verify = False
        self.mdl_file = mdl_file
        self.best_accuracy = 0
        self.AgentClass = AgentClass
        self.agent = AgentClass(
            state_size=self.maze.observation_space.shape[0],
            action_size=self.maze.action_space.n,
            mdl_file=self.mdl_file)

    def init_dbg(self):
        self.dbg_info = [[0 for i in range(self.maze.height)]
                         for j in range(self.maze.width)]

    def set_dbg_info(self):
        for w in range(self.maze.width):
            for h in range(self.maze.height):
                state = self.maze.observation(np.array([w, h]))
                _, _, info = self.agent.act_raw(state, predict_only=True)
                self.dbg_info[w][h] = info

    @staticmethod
    def explore(env, agent, quiet):
        done = False
        state = env.get_poition()
        keys = []
        cache = []
        continue_repeated_cn = 0
        while True:
            action, _ = agent.act(state)
            key, next_state, reward, done = env.step(action, quiet=quiet)
            try:
                keys.index(key)
                continue_repeated_cn += 1
                if continue_repeated_cn > 5:
                    break
            except ValueError:
                keys.append(key)
                cache.append((state, action, reward, next_state, done))
                continue_repeated_cn = 0
                if done and (reward > 0):
                    break
            state = next_state
        return done and (reward > 0), cache

    def init_run_var(self):
        self.done_t = 0
        self.lost_t = 0
        self.verify_t = 0
        self.delay_t = DISPLAY_DELAY
        self.last_steps = 0
        self.cur_episode = 0

    def run(self):
        self.init_dbg()
        self.init_run_var()
        no_verify_episode = 0
        no_find_episode = 0
        for self.cur_episode in range(self.episode):
            if not self.wait_verify:
                self.maze.reset(random.random() < min(self.agent.epsilon, 0.5))
                find_goal, cache = AbstractPlayer.explore(
                    self.maze, self.agent, self.quiet)
                for r in cache:
                    self.agent.remember(r)
                self.agent.train()
                self.agent.evaluate()
                if find_goal:
                    no_find_episode = 0
                    if self.agent.score[1] > ACCURACY_THRESHOLD:
                        self.agent.decrease_epsilon()
                else:
                    no_find_episode += 1
                    if no_find_episode > 5:
                        self.agent.increase_epsilon()
                if (find_goal and self.agent.score[1] > ACCURACY_THRESHOLD):
                    self.agent.decrease_epsilon()
            no_verify_episode += 1
            if no_verify_episode > VERIFY_EPISODE and (self.agent.score[1] >
                                                       ACCURACY_THRESHOLD):
                if self.verify():
                    self.agent.save_model()
                    break
                else:
                    no_verify_episode = 0
            if self.cur_episode % VERIFY_EPISODE == VERIFY_EPISODE - 1:
                self.statistic()
                self.set_dbg_info()
        else:
            self.verify()
            self.statistic()
        self.exit()

    def exit(self):
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
        self.lost_t = 0
        return False

    def verify_fail_proc(self, steps):
        self.last_steps = steps
        self.done_t = max(self.done_t - 1, 0)
        if self.wait_verify:
            self.wait_verify = False
            print("[{}] Out of verification status...".format(
                self.cur_episode))
            self.verify_t = 0
            self.delay_t = DISPLAY_DELAY

    def verify_experince(self, deplay_time=DISPLAY_DELAY):
        i = 0
        done = False
        state = self.maze.get_poition()
        # self.maze.render()
        track = [list(state)]
        while not done:
            i = i + 1
            action, _ = self.agent.act(state, True)
            _, state, reward, done = self.maze.step(action, quiet=False)
            if list(state) in track:
                break
            else:
                track.append(list(state))
            self.maze.render()
            time.sleep(deplay_time)
            if i > self.max_step:
                break
        return reward, done, i

    def verify(self):
        self.maze.reset()
        reward, done, steps = self.verify_experince(self.delay_t)
        if done and (reward > 0):
            return self.verify_suc_proc()
        else:
            self.verify_fail_proc(steps)
            return False

    def statistic(self):
        print("[{}] time:{:.2f}, steps:{}, epsilon:{:.3f}, "
              "loss:{:.3f}, acc:{:.3f}, train_c:{}".format(
                  self.cur_episode,
                  time.time() - self.start_t, self.last_steps,
                  self.agent.epsilon, self.agent.score[0], self.agent.score[1],
                  self.agent.train_count))
        if self.mdl_file:
            if self.best_accuracy == 0:
                self.best_accuracy = self.agent.score[1]
            if self.agent.score[1] > self.best_accuracy:
                self.agent.save_model()
                self.best_accuracy = self.best_accuracy
