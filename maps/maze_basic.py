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

GRID_SIZE = 8  # grid size
BLOCK = -1  # mark of block
GOAL = 100  # reward & mark of goal
EMPTY = 0  # reward of empty


class Maze:
    def __init__(self, grid_size=GRID_SIZE, block_num=GRID_SIZE):
        self.action_space = ['U', 'D', 'L', 'R']  # up, down, left, right
        self.n_actions = len(self.action_space)
        self.feature_space = ['x', 'y']
        self.n_features = len(self.feature_space)

        # grid size must larger than 2
        if grid_size >= 2:
            self.grid_size = grid_size
        else:
            self.grid_size = GRID_SIZE

        # the number of blocks, standard for the level of difficult
        self.n_blocks = min(
            max(block_num, 0), self.grid_size * self.grid_size - 2)
        self.n_distance = self.n_blocks * 2 + 4

        # for store the info of map
        self.r_map = [[0 for i in range(grid_size)] for j in range(grid_size)]
        self.origin_coords = np.array([int(0), int(0)])
        self.cur_coords = np.array([int(0), int(0)])
        self.goal_coords = np.array([int(0), int(0)])

        # generate the maze
        self._generate_maze()

    def _generate_maze(self):
        # clear map
        self.r_map = [[0 for i in range(self.grid_size)]
                      for j in range(self.grid_size)]

        # set goal
        w = random.randrange(self.grid_size)
        h = random.randrange(self.grid_size)
        self.r_map[w][h] = GOAL
        self.goal_coords = np.array([w, h])

        # set blocks randomly
        for _ in range(self.n_blocks):
            while 0 != self.r_map[w][h]:
                w = random.randrange(self.grid_size)
                h = random.randrange(self.grid_size)
            else:
                self.r_map[w][h] = BLOCK

        # generate me
        self.reset(r_position=True, reset_origin=True, distance=self.grid_size)

    def reset(self, r_position=False, reset_origin=False, distance=0):
        if r_position:
            w = random.randrange(self.grid_size)
            h = random.randrange(self.grid_size)
            while (EMPTY != self.r_map[w][h]) or (np.sum(
                    np.abs(np.array([w, h]) - self.goal_coords)) < distance):
                w = random.randrange(self.grid_size)
                h = random.randrange(self.grid_size)
            if reset_origin:
                self.origin_coords = np.array([w, h])
        else:
            w = self.origin_coords[0]
            h = self.origin_coords[1]

        self.cur_coords = np.array([int(w), int(h)])
        return self.get_goal_distance(self.cur_coords)

    def step(self, action, print_track=False):
        reward = EMPTY
        done = False
        r_map = self.r_map
        old = self.cur_coords.copy()

        if action == 0:  # up
            self.cur_coords[1] = max(self.cur_coords[1] - 1, 0)
        elif action == 1:  # down
            self.cur_coords[1] = min(self.cur_coords[1] + 1,
                                     self.grid_size - 1)
        elif action == 2:  # right
            self.cur_coords[0] = min(self.cur_coords[0] + 1,
                                     self.grid_size - 1)
        elif action == 3:  # left
            self.cur_coords[0] = max(self.cur_coords[0] - 1, 0)

        if BLOCK == r_map[self.cur_coords[0]][self.cur_coords[1]]:
            self.cur_coords = old.copy()
        if not (self.cur_coords - old).any():
            reward = BLOCK
        elif not (self.cur_coords - self.goal_coords).any():
            reward = GOAL * self.n_blocks
            done = True

        if print_track:
            print("track:{}->{}->{}".format(old, action, self.cur_coords))

        return self.get_goal_distance(self.cur_coords), reward, done

    def get_r_map(self):
        return self.r_map.copy()

    def get_cur_coords(self):
        return self.cur_coords.copy()

    def get_goal_coords(self):
        return self.goal_coords.copy()

    def get_goal_distance(self, coords):
        return (coords - self.goal_coords) / self.grid_size

    def get_block_distance(self, coords):
        dis_lst = np.zeros([self.n_blocks, 2])
        n = 0
        for w in range(self.grid_size):
            for h in range(self.grid_size):
                if BLOCK == self.r_map[w][h]:
                    dis_lst[
                        n][:] = (coords - np.array([w, h])) / self.grid_size
                    n += 1
        return dis_lst

    def get_wall_distacne(self, coords):
        return coords / self.grid_size - np.array([0.5, 0.5])

    def get_all_distance(self, coords=[]):
        if len(coords) != 2:
            c = self.cur_coords
        else:
            c = coords
        return np.append(
            np.append(self.get_goal_distance(c), self.get_wall_distacne(c)),
            self.get_block_distance(c)), self.r_map[c[0]][c[1]]


# simple test
'''
if __name__ == "__main__":
    env = Maze(grid_size=GRID_SIZE, n_blocks=GRID_SIZE * 2)
    for _ in range(1000):
        env.step(random.randrange(env.n_actions), print_track=True)

    # wait for confirm
    input()
'''
