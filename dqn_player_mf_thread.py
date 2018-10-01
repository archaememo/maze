import numpy as np
import time
import threading
import copy
import collections
import sys
from maps.maze_basic import Maze
from maps.maze_board import MazeBoard
from agents.ddqn_agent import DQNAgent

INC_EPISODE_THRESHOLD = 17
INC_EPISODE_RATE = 1.1


class MazePlayer(object):
    def __init__(self,
                 delay=0.01,
                 quiet=False,
                 grid_size=8,
                 block_num=8,
                 explorer_num=4):
        ''' init function of class MazePlayer'''

        print(
            "Play with parameters: delay={}, quiet={}, grid={}, block={}, explorer={}".
            format(delay, quiet, grid_size, block_num, explorer_num))
        # initilization
        self._init_variables(delay, quiet, grid_size, block_num, explorer_num)
        self._init_maze_map()
        self._init_agent()
        self._init_players()

        # enter main loop
        try:
            self.run()
        except Exception as e:
            print(e)
            self.thread_run = False
            [player.join() for player in self.players]

    def _init_variables(self, delay, quiet, grid_size, block_size,
                        explorer_num):
        self.delay_default = delay
        self.quiet = quiet
        self.grid_size = grid_size
        self.n_block = block_size
        self.n_explorer = explorer_num
        self.max_step = self.grid_size * self.grid_size
        self.episode = 50000
        self.learn_batch = self.n_block
        self.pixel_unit = 40
        self.mem_len = self.max_step * self.n_explorer * 4
        self.start_t = time.time()

    def _init_maze_map(self):
        self.maze = Maze(grid_size=self.grid_size, block_num=self.n_block)
        self.n_features = self.maze.n_distance
        self.n_actions = self.maze.n_actions

    def _init_agent(self):
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

    def _init_players(self):
        self.thread_run = True
        self.wait_verify = False
        self.players = [
            threading.Thread(
                target=self.explore_thread, args=(copy.deepcopy(self.maze), i))
            for i in range(self.n_explorer)
        ]
        self.players.append(
            threading.Thread(target=self.train_thread, args=(0, )))

    def _init_board(self):
        # init display board
        self.board = MazeBoard(self.maze.get_r_map(), unit=self.pixel_unit)
        self.board.set_origin(self.maze.cur_coords)
        self.board.set_me(self.maze.cur_coords)

    def explore_thread(self, env, i):
        done = False
        if not self.quiet:
            print("explore_thread-{} started...".format(i))
        cache = collections.deque(maxlen=self.grid_size)
        # time.sleep(np.random.rand())
        while self.thread_run:
            if ((not self.wait_verify)
                    and np.random.rand() <= self.agent.epsilon):
                _ = env.reset(r_position=True)
            else:
                _ = env.reset()
            state, _ = env.get_all_distance()
            state = np.reshape(state, [1, env.n_distance])
            for _ in range(self.max_step):
                # explore the maze
                action, _ = self.agent.act(state)
                _, reward, done = env.step(action)
                next_state, _ = env.get_all_distance()
                next_state = np.reshape(next_state, [1, env.n_distance])

                # Remember the previous state, action, reward, and done
                self.agent.remember((state, action, reward, next_state, done))

                # cache the last status to enforce remember
                cache.append((state, action, reward, next_state, done))

                if done:
                    # enforce success memory
                    for _ in range(len(cache)):
                        data = cache.pop()
                        for _ in range(self.grid_size):
                            self.agent.remember(data)
                    break
                state = next_state
            # keras seem not good to prcess multi-thread
            time.sleep(np.random.rand())
        # else:
        #   print("player_thread-{} stopped...".format(i))

    def train_thread(self, i):
        if not self.quiet:
            print("train_thread started...")
        # time.sleep(np.random.rand())
        while (self.thread_run):
            # stop training to verify the result
            if self.wait_verify:
                time.sleep(np.random.rand())
            else:
                self.agent.train(
                    batch_size=self.learn_batch, mod_epsilon=False)
        # else:
        #   print("train_thread stopped...")

    def run(self):
        self.done_t = 0
        self.lost_t = 0
        self.verify_t = 0
        self.delay_t = self.delay_default
        self.last_steps = 0
        self.cur_episode = 0

        # start all players, include explore and train
        for player in self.players:
            player.start()
            time.sleep(1)

        if not self.quiet:
            self._init_board()
            verify_exp = self.verify_exp
        else:
            verify_exp = self.verify_exp_quiet

        # let explore and train threads run for a while
        for n in range(5):
            if not self.quiet:
                self.board.render()
                self.board.set_text(self.maze.cur_coords, "{}".format(5 - n))
            time.sleep(0.5)

        for self.cur_episode in range(self.episode):
            time.sleep(self.delay_t)
            done, steps = verify_exp(self.delay_t)
            if done:
                if self.verify_suc_proc():
                    break
            else:
                self.verify_fail_proc(steps)
            # descrease esiilon each round
            self.agent.decrease_epsilon()
            if (not self.quiet) and self.cur_episode % 20 == 19:
                self.statistic()

        # game over
        self.thread_run = False
        self.statistic()
        verify_exp(self.delay_t, color="green", print_track=True)
        [player.join() for player in self.players]
        time.sleep(5)

    def verify_suc_proc(self):
        if self.wait_verify:
            self.verify_t += 1
            if self.verify_t > 3:
                print("[{}] success to train mode to find goal...".format(
                    self.cur_episode))
                return True
        else:
            self.done_t += 1
            if self.done_t == 3:
                self.wait_verify = True
                print("[{}] Enter into verification status...".format(
                    self.cur_episode))
                self.delay_t = 0.5
                self.done_t = 0
        self.agent.modify_epsilon(0.5)
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

    def verify_exp(self, deplay_time=0.01, color="red", print_track=False):
        i = 0
        done = False
        self.maze.reset()
        self.board.clear_text()
        self.board.render()
        state, _ = self.maze.get_all_distance()
        self.board.set_me(self.maze.cur_coords)
        track = [list(state)]
        while not done:
            i = i + 1
            state = np.reshape(state, [1, self.n_features])
            old_coords = self.maze.cur_coords.copy()
            action, _ = self.agent.act(state, True)
            _, _, done = self.maze.step(action, print_track=print_track)
            state, _ = self.maze.get_all_distance()
            if list(state) in track:
                break
            else:
                track.append(list(state))
            # refresh board dispalyer
            self.board.set_me(self.maze.cur_coords)
            self.board.set_text(old_coords, "X", color=color)
            self.board.render()
            # have to wait for a while to avoid keras crash
            time.sleep(deplay_time)
            if i > self.max_step:
                break
        return done, i

    def verify_exp_quiet(self, deplay_time=0.1, color="red",
                         print_track=False):
        i = 0
        done = False
        _ = self.maze.reset()
        state, _ = self.maze.get_all_distance()
        track = [list(state)]
        while not done:
            i = i + 1
            state = np.reshape(state, [1, self.n_features])
            action, _ = self.agent.act(state, True)
            _, _, done = self.maze.step(action, print_track=print_track)
            state, _ = self.maze.get_all_distance()
            if list(state) in track:
                break
            else:
                track.append(list(state))
            if i > self.max_step:
                break
            # have to wait for a while to avoid keras crash
            time.sleep(deplay_time)
        return done, i

    def display_q(self, display_type):
        ''' dispaly q value in the borad '''
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

    def statistic(self):
        print(
            "[{}] time:{:.2f}, steps:{}, epsilon:{:.3f}, "
            "score:{:.3f}, train_c:{}".format(
                self.cur_episode,
                time.time() - self.start_t, self.last_steps,
                self.agent.epsilon, self.agent.score, self.agent.train_count))


def main():
    default_params = [float(0.01), False, int(16), int(16), int(4)]
    analysis = [
        lambda x: float(x), lambda x: x in ["True", "true", "TRUE"],
        lambda x: int(x), lambda x: int(x), lambda x: int(x)
    ]
    params = default_params[:]
    try:
        for i in range(min(len(params), len(sys.argv) - 1)):
            params[i] = list(map(analysis[i], [default_params[i]]))[0]
    except Exception as e:
        print(e)
        print("going to start with default configration")
        params = default_params[:]

    MazePlayer(
        delay=params[0],
        quiet=params[1],
        grid_size=params[2],
        block_num=params[3],
        explorer_num=params[4])


if __name__ == "__main__":
    main()
