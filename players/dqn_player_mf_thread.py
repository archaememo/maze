import numpy as np
import time
import threading
import copy
import random
import collections
import sys
sys.path.append('.')
from maps.maze_basic import Maze  # flake8: noqa
from maps.maze_board import MazeBoard  # flake8: noqa
from agents.ddqn_agent import DQNAgent  # flake8: noqa



class MazePlayer(object):
    def __init__(self):
        #constant
        self.grid_size = 16
        self.n_block = self.grid_size
        self.max_step = self.grid_size * self.grid_size
        self.episode = 5000
        self.learn_batch = self.n_block
        self.n_player = 8
        self.pixel_unit = 40
        self.mem_len = self.max_step * self.n_player * 4
        self.maze = Maze(grid_size=self.grid_size, block_num=self.n_block)
        self.n_features = self.maze.n_distance
        self.n_actions = self.maze.n_actions
        self.agent = DQNAgent(
            self.n_features,
            self.n_actions,
            mem_len=self.mem_len,
            layer_info=[
                self.grid_size * self.grid_size * self.n_block,
                self.grid_size * self.grid_size * self.n_block *
                self.n_actions,
                self.grid_size * self.grid_size * self.n_actions,
            ])
        self.thread_run = True
        self.wait_verify = False
        self.players = [
            threading.Thread(
                target=self.player_thread, args=(copy.deepcopy(self.maze), i))
            for i in range(self.n_player)
        ]
        self.players.append(
            threading.Thread(target=self.train_thread, args=(0, )))
        try: 
            self.board = MazeBoard(self.maze.get_r_map(), unit=self.pixel_unit)
            self.board.set_origin(self.maze.cur_coords)
            self.board.set_me(self.maze.cur_coords)
            self.run()
        except:
            self.thread_run= False
            [player.join() for player in self.players]

        # self.board.after(100, self.run, 1)
        # self.board.mainloop()

    def player_thread(self, env, i):
        done = False
        e = 0
        print("player_thread-{} started...".format(i))
        cache = collections.deque(maxlen=self.grid_size)
        time.sleep(random.random())
        while (self.thread_run):
            e += 1
            n_rpos = (1 != e % 2)
            if n_rpos:
                _ = env.reset()
            else:
                _ = env.reset(r_position=True)
            state, _ = env.get_all_distance()
            state = np.reshape(state, [1, env.n_distance])
            for _ in range(self.max_step):
                # Decide action
                action, _ = self.agent.act(state)
                _, reward, done = env.step(action)
                next_state, _ = env.get_all_distance()
                next_state = np.reshape(next_state, [1, env.n_distance])
                # Remember the previous state, action, reward, and done
                self.agent.remember((state, action, reward, next_state, done))
                cache.append((state, action, reward, next_state, done))
                # make next_state the new current state for the next frame.
                if done:
                    for _ in range(len(cache)):
                        data = cache.pop()
                        for _ in range(self.grid_size):
                            self.agent.remember(data)
                    break
                state = next_state
            time.sleep(random.random())
            # print("player_thread:{},round:{}".format(i,e))
            # time.sleep(0)
        else:
            print("player_thread-{} stopped...".format(i))

    def train_thread(self, i):
        print("train_thread started...")
        time.sleep(random.random())
        while (self.thread_run):
            if self.wait_verify:
                time.sleep(random.random())
            else:
                self.agent.train(
                    batch_size=self.learn_batch, mod_epsilon=False)
        else:
            print("train_thread stopped...")
            sys.stdout.flush()

    def verify_exp(self, deplay_time=0.1, color="red"):
        i = 0
        done = False
        _ = self.maze.reset(dis=self.grid_size)
        state, _ = self.maze.get_all_distance()
        self.board.set_me(self.maze.cur_coords)
        track = [list(state)]
        while not done:
            i = i + 1
            state = np.reshape(state, [1, self.n_features])
            old_coords = self.maze.cur_coords.copy()
            action, _ = self.agent.act(state, True)
            _, _, done = self.maze.step(action, print_track=False)
            state, _ = self.maze.get_all_distance()
            if list(state) in track:
                break
            else:
                track.append(list(state))
            self.board.set_me(self.maze.cur_coords)
            self.board.set_text(old_coords, "X", color=color)
            self.board.render()
            time.sleep(deplay_time)
            if i > self.max_step:
                break
        return done, i

    def verify_exp_quiet(self):
        i = 0
        done = False
        _ = self.maze.reset(dis=self.grid_size)
        state, _ = self.maze.get_all_distance()
        track = [list(state)]
        while not done:
            i = i + 1
            state = np.reshape(state, [1, self.n_features])
            # old_coords = self.maze.cur_coords.copy()
            action, _ = self.agent.act(state, True)
            _, _, done = self.maze.step(action, print_track=False)
            state, _ = self.maze.get_all_distance()
            if list(state) in track:
                break
            else:
                track.append(list(state))
            if i > self.max_step:
                break
        return done, i

    def display_q(self, display_type):
        for w in range(self.grid_size):
            for h in range(self.grid_size):
                coords = np.array([w, h])
                state, reward = self.maze.get_all_distance(coords)
                if reward != 0:
                    continue
                state = np.reshape(state, [1, self.n_features])
                action, _, raw_action = self.agent.act_raw(state, True)
                if 2 == display_type:
                    self.board.set_text(
                        coords,
                        "{}:{:+.2f}\n{}:{:+.2f}\n{}:{:+.2f}\n{}:{:+.2f}".
                        format(self.maze.action_space[0], raw_action[0],
                               self.maze.action_space[1], raw_action[1],
                               self.maze.action_space[2], raw_action[2],
                               self.maze.action_space[3], raw_action[3]))
                else:
                    self.board.set_text(coords, self.maze.action_space[action])
                self.board.render()

    def run(self, display_type=0):
        last_steps = 0
        lost_t = 0
        done_t = 0
        verify_t = 0
        delay_t = 0
        [player.start() for player in self.players]
        for n in range(10):
            self.board.render()
            # wait for training
            self.board.set_text(self.maze.cur_coords, "{}".format(10-n))
            time.sleep(0.5)
        for _ in range(self.episode):
            time.sleep(0)
            self.board.clear_text()
            self.board.render()
            done, steps = self.verify_exp(delay_t)
            if done:
                if self.wait_verify:
                    verify_t += 1
                    if verify_t > 2:
                        break
                else:
                    done_t += 1
                    if done_t == 3:
                        self.wait_verify = True
                        delay_t = 0.5
                        done_t = 0
                self.agent.modify_epsilon(0.5)
                lost_t = 0
            else:
                if steps <= last_steps:
                    lost_t += 1
                    if lost_t > 18:
                        self.agent.modify_epsilon(1.1)
                        lost_t = 0
                else:
                    lost_t = max(lost_t - 1, 0)
                last_steps = steps
                done_t = max(done_t - 1, 0)
                if self.wait_verify:
                    self.wait_verify = False
                    verify_t = 0
                    delay_t = 0
            if display_type > 0:
                self.display_q(display_type)
            self.agent.decrease_epsilon()
            self.board.render()
        self.thread_run = False
        [player.join() for player in self.players]
        self.board.clear_text()
        self.board.render()
        self.verify_exp(delay_t, "green")
        input()

if __name__ == "__main__":
    MazePlayer()
