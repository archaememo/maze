"""
Rer_maprcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       blocks           [reward = -1].
Yellow bin circle:      goal paradise    [reward = +1].
The edge                walls            [reward = -1].
All other states:       ground           [reward = 0].
This script is the environment part of this example.
Forked from https://morvanzhou.github.io/tutorials/ and
make some improvement
"""

import numpy as np
import random
import gym
import gym.spaces as spaces
# import os
import envs.board as board
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# Hyper parameter
# ==============================================================
HEIGHT = 8
WIDTH = 8
N_BLOCK = 8
BLOCK = -1  # mark of block
GOAL = 100  # reward & mark of goal
EMPTY = 0  # reward of empty
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
UNIT = 40  # default pixels of a grid
# ==============================================================


class Maze(board.Board, gym.Env):
    def __init__(self, height=HEIGHT, width=WIDTH, n_block=N_BLOCK, unit=UNIT):
        # action ['Up', 'Down', 'Left', 'Right']
        super(Maze, self).__init__(height + 2, width + 2, unit)
        self.n_block = n_block
        # self._generate_maze()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            np.array([0, 0, BLOCK, BLOCK, BLOCK, BLOCK]),
            np.array([width, height, EMPTY, EMPTY, EMPTY, EMPTY],
                     dtype=np.int64))

    def generate_maze(self):
        # clear map
        self.maze_map = [[0 for i in range(self.width)]
                         for j in range(self.height)]

        # set edge
        for w in range(self.width):
            self.maze_map[w][0] = BLOCK
            self.maze_map[w][self.height - 1] = BLOCK
        for h in range(1, self.height - 1):
            self.maze_map[0][h] = BLOCK
            self.maze_map[self.width - 1][h] = BLOCK
        self.init_window()

        # set goal
        w = random.randrange(self.width)
        h = random.randrange(self.height)
        while 0 != self.maze_map[w][h]:
            w = random.randrange(self.width)
            h = random.randrange(self.height)
        self.maze_map[w][h] = GOAL
        self.goal_position = np.array([w, h])
        self.set_chess(w, h, "oval", "yellow")

        # set blocks randomly
        for _ in range(self.n_block):
            while 0 != self.maze_map[w][h]:
                h = random.randrange(self.height)
                w = random.randrange(self.width)
            else:
                self.maze_map[w][h] = BLOCK
                self.set_chess(w, h, "rectangle", "black")

        # set me
        w = random.randrange(self.width)
        h = random.randrange(self.height)
        while (EMPTY != self.maze_map[w][h]) or (np.sum(
                np.abs(np.array([w, h]) - self.goal_position)) < int(
                    (self.height + self.width - 4) / 2)):
            # make me at least (self.height+self.width)/2 away from goal
            h = random.randrange(self.height)
            w = random.randrange(self.width)
        self.origin_position = np.array([w, h])
        self.cur_position = self.origin_position.copy()
        self.carvas_me = self.set_chess(w, h, "rectangle", "green")

    def reset(self):
        self.cur_position = self.origin_position.copy()
        self.move_chess(self.carvas_me, self.cur_position[0],
                        self.cur_position[1])
        self.clear_all_text()
        return self.observation(self.cur_position)

    def step(self, action, quiet=False):
        reward = EMPTY
        maze_map = self.maze_map
        position = self.cur_position.copy()

        if action == UP:  # up
            position[1] = max(self.cur_position[1] - 1, 0)
        elif action == DOWN:  # down
            position[1] = min(self.cur_position[1] + 1, self.height - 1)
        elif action == LEFT:  # left
            position[0] = max(self.cur_position[0] - 1, 0)
        elif action == RIGHT:  # right
            position[0] = min(self.cur_position[0] + 1, self.width - 1)
        reward = maze_map[position[0]][position[1]]

        # if not quiet:
        #     print("track:{}->{}->{}:{}".format(self.cur_position, action,
        #                                        position, reward))

        if BLOCK != reward:
            if not quiet:
                self.move_chess(self.carvas_me, position[0], position[1])
                self._set_trace(self.cur_position)
                self.render()
            self.cur_position = position.copy()

        return self.observation(position), reward, reward != EMPTY

    def _set_trace(self, position, txt="X"):
        if not (self.origin_position - position).any():
            fill_color = 'purple'
        else:
            fill_color = 'gray'
        self.set_text(position[0], position[1], "X", fill_color)

    def observation(self, position):
        distance = position - self.goal_position
        next_state = np.array([0, 0, 0, 0])
        if (position[1] == 0) or (self.maze_map[position[0]][position[1] -
                                                             1] == BLOCK):
            next_state[0] = -1
        if (position[1] == self.height -
                1) or (self.maze_map[position[0]][position[1] + 1] == BLOCK):
            next_state[1] = -1
        if (position[0] == 0) or (self.maze_map[position[0] -
                                                1][position[1]] == BLOCK):
            next_state[2] = -1
        if (position[0] == self.width -
                1) or (self.maze_map[position[0] + 1][position[1]] == BLOCK):
            next_state[3] = -1
        # return np.append(np.append(self.cur_position, distance), next_state)
        return np.append(distance, next_state)

    def save_map(self, path):
        with open(path, "w+") as f:
            f.write(str(self.width))
            f.write("\n")
            f.write(str(self.height))
            f.write("\n")
            f.write(str(self.unit))
            f.write("\n")
            f.write("{},{}".format(self.origin_position[0],
                                   self.origin_position[1]))
            f.write("\n")
            for i in range(self.width):
                for j in range(self.height):
                    f.write("{},".format(self.maze_map[i][j]))
                f.seek(f.tell() - 1)
                f.write("\n")

    def load_map(self, path):
        with open(path, "r") as f:
            self.width = int(f.readline())
            self.height = int(f.readline())
            self.unit = int(f.readline())
            self.init_window()

            origin_position = f.readline().split(",")
            self.origin_position[0] = int(origin_position[0])
            self.origin_position[1] = int(origin_position[1])
            self.cur_position = self.origin_position.copy()
            self.set_chess(self.origin_position[0], self.origin_position[1],
                           "rectangle", "green")
            self.maze_map = [[0 for i in range(self.width)]
                             for j in range(self.height)]
            self.n_block = 0
            for w in range(self.width):
                grids = f.readline().split(",")
                for h in range(self.height):
                    self.maze_map[w][h] = int(grids[h])
                    if self.maze_map[w][h] == BLOCK:
                        self.n_block += 1
                        self.set_chess(w, h, "rectangle", "black")
                    elif self.maze_map[w][h] == GOAL:
                        self.goal_position[0] = w
                        self.goal_position[1] = h
                        self.set_chess(w, h, "oval", "yellow")

    # def _move_me(self):
    #     old_position = np.array([(int(self.canvas.coords(self.carvas_me)[0])),
    #                              (int(self.canvas.coords(self.carvas_me)[1]))
    #                              ]) // self.unit
    #     if (self.cur_position - old_position).any():
    #         self.canvas.move(
    #             self.carvas_me,
    #             self.unit * (self.cur_position[0] - old_position[0]),
    #             self.unit * (self.cur_position[1] - old_position[1]))
    #         self._set_trace(old_position)

    # def _clear_text(self):
    #     for w in range(self.width):
    #         for h in range(self.height):
    #             if 0 != self.canvas_map[w][h]:
    #                 if self.canvas.gettags(self.canvas_map[w][h])[0] == 'text':
    #                     self.canvas.itemconfig(self.canvas_map[w][h], text="")

    # def render(self):
    #     self.update()


# simple test
if __name__ == "__main__":
    env = Maze()
    env.generate_maze()
    # env.mainloop()
    for i in range(100000):
        # debug
        env.render()
        env.step(random.randrange(env.action_space.n), quiet=False)
        env.render()
        if i % 100 == 0:
            env.reset()
    input()
