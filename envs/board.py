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
from config import UNIT
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class Board(tk.Tk, object):
    def __init__(self, width, height):
        # action ['Up', 'Down', 'Left', 'Right']
        super(Board, self).__init__()
        self.height = height
        self.width = width
        self._init_window()

    def reset(self):
        self.cur_position = self.origin_position.copy()
        self._clear_text()
        return self.get_observation

    def _init_window(self):
        # width x height
        self.geometry('{0}x{1}'.format((self.width - 1) * UNIT,
                                       (self.height - 1) * UNIT))

        # canvas map of maze
        self.canvas_map = [[0 for i in range(self.width)]
                           for j in range(self.height)]
        self.canvas_text = [[0 for i in range(self.width)]
                            for j in range(self.height)]

        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=(self.height - 1) * UNIT,
                                width=(self.width - 1) * UNIT)

        cv = self.canvas

        # create grids
        cv.create_line(0, 0, 0, (self.height - 1) * UNIT)
        cv.create_line(UNIT // 2, 0, UNIT // 2, (self.height - 1) * UNIT)
        cv.create_line(int((self.width - 1.5) * UNIT), 0,
                       int((self.width - 1.5) * UNIT),
                       (self.height - 1) * UNIT)
        cv.create_line(int((self.width - 1) * UNIT), 0,
                       int((self.width - 1) * UNIT), (self.height - 1) * UNIT)

        cv.create_line(0, 0, (self.width - 1) * UNIT, 0)
        cv.create_line(0, UNIT // 2, (self.width - 1) * UNIT, UNIT // 2)
        cv.create_line(0, int(
            (self.height - 1.5) * UNIT), (self.width - 1) * UNIT,
                       int((self.height - 1.5) * UNIT))
        cv.create_line(0, int((self.height - 1) * UNIT),
                       (self.width - 1) * UNIT, int((self.height - 1) * UNIT))

        for c in range(UNIT // 2, self.height * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.height * UNIT
            cv.create_line(x0, y0, x1, y1)
        for r in range(UNIT // 2, self.width * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.width * UNIT, r
            cv.create_line(x0, y0, x1, y1)

        self._set_edge()
        self.canvas.pack()

    def _set_edge(self):
        self.canvas.create_rectangle(0,
                                     0,
                                     UNIT // 2,
                                     UNIT // 2,
                                     fill="gray",
                                     tags="edge")
        self.canvas.create_rectangle(int((self.width - 1.5) * UNIT),
                                     0,
                                     int((self.width - 1) * UNIT),
                                     UNIT // 2,
                                     fill="gray",
                                     tags="edge")
        self.canvas.create_rectangle(0,
                                     int((self.height - 1.5) * UNIT),
                                     UNIT // 2,
                                     int((self.height - 1) * UNIT),
                                     fill="gray",
                                     tags="edge")
        self.canvas.create_rectangle(int((self.width - 1.5) * UNIT),
                                     int((self.height - 1.5) * UNIT),
                                     int((self.width - 1) * UNIT),
                                     int((self.height - 1) * UNIT),
                                     fill="gray",
                                     tags="edge")

        for x in range(UNIT // 2, (self.width - 2) * UNIT, UNIT):
            self.canvas.create_rectangle(x,
                                         0,
                                         x + UNIT,
                                         UNIT // 2,
                                         fill="gray",
                                         tags="edge")
            self.canvas.create_rectangle(x,
                                         int((self.height - 1.5) * UNIT),
                                         x + UNIT,
                                         int((self.height - 1) * UNIT),
                                         fill="gray",
                                         tags="edge")
        for y in range(UNIT // 2, (self.height - 2) * UNIT, UNIT):
            self.canvas.create_rectangle(0,
                                         y,
                                         UNIT // 2,
                                         y + UNIT,
                                         fill="gray",
                                         tags="edge")
            self.canvas.create_rectangle(int((self.width - 1.5) * UNIT),
                                         y,
                                         int((self.width - 1) * UNIT),
                                         y + UNIT,
                                         fill="gray",
                                         tags="edge")

    def set_chess(self, x, y, shape, color, tags="chess"):
        shape_funcs = {
            "rectangle": self.canvas.create_rectangle,
            "oval": self.canvas.create_oval
        }
        func = shape_funcs[shape]

        if 0 != self.canvas_map[x][y]:
            self.canvas.delete(self.canvas_map[x][y])

        self.canvas_map[x][y] = func(x * UNIT - UNIT * 3 / 8,
                                     y * UNIT - UNIT * 3 / 8,
                                     x * UNIT + UNIT * 3 / 8,
                                     y * UNIT + UNIT * 3 / 8,
                                     fill=color,
                                     tags=tags)
        return self.canvas_map[x][y]

    def move_chess(self, carvas_obj, x, y):
        old_x = int((self.canvas.coords(carvas_obj)[0] + UNIT / 2) / UNIT)
        old_y = int((self.canvas.coords(carvas_obj)[1] + UNIT / 2) / UNIT)
        if (x - old_x) or (y - old_y):
            self.canvas.move(carvas_obj, UNIT * (x - old_x),
                             UNIT * (y - old_y))

    def set_text(self, x, y, txt="X", color="gray"):
        if 0 != self.canvas_text[x][y]:
            self.canvas.itemconfig(self.canvas_text[x][y],
                                   text=txt,
                                   fill=color,
                                   tags="text")
        else:
            self.canvas_text[x][y] = self.canvas.create_text(x * UNIT,
                                                             y * UNIT,
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
    pass
