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

# Hyper parameter
# ==============================================================
WIDTH = 8  # axis-x
HEIGHT = 8  # axis-y
N_BLOCK = 8
BLOCK = -1  # reward & mark of block
GOAL = 100  # reward & mark of goal
EMPTY = 0  # reward & mark of empty
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
# ==============================================================


class Maze(gym.Env):
    def __init__(self, height=HEIGHT, width=WIDTH, n_block=N_BLOCK):
        # action ['Up', 'Down', 'Left', 'Right']
        self.height = height + 2  # Reserve 2 unit for up wall and down wall
        self.width = width + 2  # Reserve 2 unit for left wall and right wall
        self.n_block = n_block
        self.origin_position = np.array([0, 0])  # start point
        self.goal_position = np.array([0, 0])
        self.action_space = spaces.Discrete(4)
        # [distance to goal in x-axis, distance to goal in y-axis,
        # reward of up, reward of down, reward of left, reward of right]
        self.observation_space = spaces.Box(
            np.array([0, 0, BLOCK, BLOCK, BLOCK, BLOCK]),
            np.array([width, height, EMPTY, EMPTY, EMPTY, EMPTY],
                     dtype=np.int64))

    def generate_maze(self):
        # create a empty map with given width and height
        self.maze_map = [[0 for i in range(self.width)]
                         for j in range(self.height)]
        # set edge wall, take edge wall as blocks
        for w in range(self.width):
            self.maze_map[w][0] = BLOCK
            self.maze_map[w][self.height - 1] = BLOCK
        for h in range(1, self.height - 1):
            self.maze_map[0][h] = BLOCK
            self.maze_map[self.width - 1][h] = BLOCK

        # set goal randomly
        w = random.randrange(self.width)
        h = random.randrange(self.height)
        # find the empty positoin
        while 0 != self.maze_map[w][h]:
            w = random.randrange(self.width)
            h = random.randrange(self.height)
        self.maze_map[w][h] = GOAL
        self.goal_position = np.array([w, h])

        # set blocks randomly
        for _ in range(self.n_block):
            # find the empty positoin
            while 0 != self.maze_map[w][h]:
                h = random.randrange(self.height)
                w = random.randrange(self.width)
            else:
                self.maze_map[w][h] = BLOCK

        # set me
        w = random.randrange(self.width)
        h = random.randrange(self.height)
        # make me at a certain distance from goal
        while (EMPTY != self.maze_map[w][h]) or (np.sum(
                np.abs(np.array([w, h]) - self.goal_position)) < int(
                    (self.height + self.width - 4) / 2)):
            h = random.randrange(self.height)
            w = random.randrange(self.width)
        self.origin_position = np.array([w, h])
        self.cur_position = self.origin_position.copy()

    def reset(self):
        self.cur_position = self.origin_position.copy()
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

        if not quiet:
            print("track:{}->{}->{}:{}".format(self.cur_position, action,
                                               position, reward))

        if BLOCK != reward:
            self.cur_position = position.copy()

        return self.observation(position), reward, reward != EMPTY

    def render(self):
        pass

    def get_poition(self):
        return self.observation(self.cur_position)

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
            origin_position = f.readline().split(",")

            self.origin_position[0] = int(origin_position[0])
            self.origin_position[1] = int(origin_position[1])
            self.cur_position = self.origin_position.copy()
            self.maze_map = [[0 for i in range(self.width)]
                             for j in range(self.height)]
            self.n_block = 0
            for w in range(self.width):
                grids = f.readline().split(",")
                for h in range(self.height):
                    self.maze_map[w][h] = int(grids[h])
                    if self.maze_map[w][h] == BLOCK:
                        if w in (0, self.width) or h in (0, self.height):
                            continue
                        self.n_block += 1
                    elif self.maze_map[w][h] == GOAL:
                        self.goal_position[0] = w
                        self.goal_position[1] = h


# simple test
if __name__ == "__main__":
    env = Maze()
    env.generate_maze()

    for i in range(100000):
        # debug
        env.step(random.randrange(env.action_space.n), quiet=False)
        if i % 100 == 0:
            env.reset()
    input()
