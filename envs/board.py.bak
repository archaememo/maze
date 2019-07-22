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
# import numpy as np
# import random
import os
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


class Board(tk.Tk, object):
    def __init__(self, height=HEIGHT, width=WIDTH, unit=UNIT, map_path=None):
        # action ['Up', 'Down', 'Left', 'Right']
        super(Board, self).__init__()
        self.unit = unit
        if map_path is not None and os.path.exists(map_path):
            self._load_map(map_path)
        else:
            self.height = height
            self.width = width
        # self._init_window()

    def reset(self):
        self.cur_position = self.origin_position.copy()
        self._clear_text()
        return self.get_observation

    def init_window(self):
        unit = self.unit
        # width x height
        self.geometry('{0}x{1}'.format(self.width * unit, self.height * unit))

        # canvas map of maze
        self.canvas_map = [[0 for i in range(self.width)]
                           for j in range(self.height)]
        self.canvas_text = [[0 for i in range(self.width)]
                            for j in range(self.height)]

        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=self.height * unit,
                                width=self.width * unit)

        cv = self.canvas

        # create grids
        for c in range(0, (self.height + 1) * unit, unit):
            x0, y0, x1, y1 = c, 0, c, self.height * unit
            cv.create_line(x0, y0, x1, y1)
        for r in range(0, (self.width + 1) * unit, unit):
            x0, y0, x1, y1 = 0, r, self.width * unit, r
            cv.create_line(x0, y0, x1, y1)
        self._set_edge()
        self.canvas.pack()



    def _set_edge(self):
        unit = self.unit
        for x in range(self.width):
            self.canvas.create_rectangle(x * unit,
                                         0,
                                         x * unit + unit,
                                         unit,
                                         fill="gray",
                                         tags="edge")
            self.canvas.create_rectangle(x * unit,
                                         (self.height - 1) * unit,
                                         x * unit + unit,
                                         self.height * unit,
                                         fill="gray",
                                         tags="edge")
        for y in range(1, self.height - 1):
            self.canvas.create_rectangle(0,
                                         y * unit,
                                         unit,
                                         y * unit + unit,
                                         fill="gray",
                                         tags="edge")
            self.canvas.create_rectangle((self.width - 1) * unit,
                                         y * unit,
                                         self.width * unit,
                                         y * unit + unit,
                                         fill="gray",
                                         tags="edge")

    def set_chess(self, x, y, shape, color, tags="chess"):
        unit = self.unit
        shape_funcs = {
            "rectangle": self.canvas.create_rectangle,
            "oval": self.canvas.create_oval
        }
        func = shape_funcs[shape]

        if 0 != self.canvas_map[x][y]:
            self.carvas.delete(self.canvas_map[x][y])

        self.canvas_map[x][y] = func(x * unit + unit / 8,
                                     y * unit + unit / 8,
                                     x * unit + unit * 7 / 8,
                                     y * unit + unit * 7 / 8,
                                     fill=color,
                                     tags=tags)
        return self.canvas_map[x][y]

    def move_chess(self, carvas_obj, x, y):
        old_x = int(self.canvas.coords(carvas_obj)[0]) // self.unit
        old_y = int(self.canvas.coords(carvas_obj)[1]) // self.unit
        if (x - old_x) or (y - old_y):
            self.canvas.move(carvas_obj, self.unit * (x - old_x),
                             self.unit * (y - old_y))

    def set_text(self, x, y, txt="X", color="gray"):
        unit = self.unit
        if 0 != self.canvas_text[x][y]:
            self.canvas.itemconfig(self.canvas_text[x][y],
                                   text=txt,
                                   fill=color,
                                   tags="text")
        else:
            self.canvas_text[x][y] = self.canvas.create_text(
                x * unit + unit / 2,
                y * unit + unit / 2,
                fill=color,
                text=txt,
                tags='text')

    def clear_text(self, x, y):
        if 0 != self.canvas_text[x][y]:
            self.canvas.itemconfig(self.canvas_text[x][y], text="")

    def clear_all_text(self):
        for x in range(self.width):
            for y in range(self.height):
                self.clear_text(x, y)

    def render(self):
        self.update()


# simple test
if __name__ == "__main__":
    env = Board()
    env.init_window()
    env.mainloop()
    # env.mainloop()
    # for i in range(100000):
    #     env.step(random.randrange(env.action_space.n), print_track=True)
    #     env.render()
    #     if i % 100 == 0:
    #         env.reset()
    input()
