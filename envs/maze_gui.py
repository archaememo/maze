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

import gym
import envs.board as board
import envs.maze as maze
from config import GOAL, BLOCK, HEIGHT, WIDTH, N_BLOCK


class MazeGUI(maze.Maze, gym.Env):
    def __init__(self, height=HEIGHT, width=WIDTH, n_block=N_BLOCK):
        super(MazeGUI, self).__init__(height, width, n_block)

    def generate_maze(self):
        super(MazeGUI, self).generate_maze()
        self._init_board()

    def reset(self, random_position=False):
        state = super(MazeGUI, self).reset(random_position)
        self.board.move_chess(self.carvas_me, self.cur_position[0],
                              self.cur_position[1])
        self.board.clear_all_text()
        return state

    def step(self, action, quiet=False):
        old_position = self.cur_position.copy()
        state = super(MazeGUI, self).step(action, quiet=True)
        if (old_position - self.cur_position).any() and not quiet:
            self.board.move_chess(self.carvas_me, self.cur_position[0],
                                  self.cur_position[1])
            self.set_trace(old_position)
            self.board.render()
        return state

    def set_trace(self, position):
        if not (self.origin_position - position).any():
            fill_color = 'purple'
        else:
            fill_color = 'gray'
        self.board.set_text(position[0], position[1], "X", fill_color)

    def render(self):
        self.board.render()

    def load_map(self, path):
        super(MazeGUI, self).load_map(path)
        self._init_board()
        return self

    def _init_board(self):
        self.board = board.Board(self.width, self.height)
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if self.maze_map[x][y] == BLOCK:
                    self.board.set_chess(x, y, "rectangle", "black")
                elif self.maze_map[x][y] == GOAL:
                    self.board.set_chess(x, y, "oval", "yellow")
        self.carvas_me = self.board.set_chess(self.cur_position[0],
                                              self.cur_position[1],
                                              "rectangle", "green")
        self.board.render()


# simple test
if __name__ == "__main__":
    pass
