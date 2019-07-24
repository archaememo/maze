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
    def __init__(self, width=WIDTH, height=HEIGHT, unit=UNIT):
        # action ['Up', 'Down', 'Left', 'Right']
        super(Board, self).__init__()
        self.unit = unit
        self.height = height
        self.width = width
        self._init_window()

    def reset(self):
        self.cur_position = self.origin_position.copy()
        self._clear_text()
        return self.get_observation

    def _init_window(self):
        unit = self.unit
        # width x height
        self.geometry('{0}x{1}'.format((self.width - 1) * unit,
                                       (self.height - 1) * unit))

        # canvas map of maze
        self.canvas_map = [[0 for i in range(self.width)]
                           for j in range(self.height)]
        self.canvas_text = [[0 for i in range(self.width)]
                            for j in range(self.height)]

        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=(self.height - 1) * unit,
                                width=(self.width - 1) * unit)

        cv = self.canvas

        # create grids
        cv.create_line(0, 0, 0, (self.height - 1) * unit)
        cv.create_line(unit // 2, 0, unit // 2, (self.height - 1) * unit)
        cv.create_line(int((self.width - 1.5) * unit), 0,
                       int((self.width - 1.5) * unit),
                       (self.height - 1) * unit)
        cv.create_line(int((self.width - 1) * unit), 0,
                       int((self.width - 1) * unit), (self.height - 1) * unit)

        cv.create_line(0, 0, (self.width - 1) * unit, 0)
        cv.create_line(0, unit // 2, (self.width - 1) * unit, unit // 2)
        cv.create_line(0, int(
            (self.height - 1.5) * unit), (self.width - 1) * unit,
                       int((self.height - 1.5) * unit))
        cv.create_line(0, int((self.height - 1) * unit),
                       (self.width - 1) * unit, int((self.height - 1) * unit))

        for c in range(unit // 2, self.height * unit, unit):
            x0, y0, x1, y1 = c, 0, c, self.height * unit
            cv.create_line(x0, y0, x1, y1)
        for r in range(unit // 2, self.width * unit, unit):
            x0, y0, x1, y1 = 0, r, self.width * unit, r
            cv.create_line(x0, y0, x1, y1)

        self._set_edge()
        self.canvas.pack()

    def _set_edge(self):
        unit = self.unit
        self.canvas.create_rectangle(0,
                                     0,
                                     unit // 2,
                                     unit // 2,
                                     fill="gray",
                                     tags="edge")
        self.canvas.create_rectangle(int((self.width - 1.5) * unit),
                                     0,
                                     int((self.width - 1) * unit),
                                     unit // 2,
                                     fill="gray",
                                     tags="edge")
        self.canvas.create_rectangle(0,
                                     int((self.height - 1.5) * unit),
                                     unit // 2,
                                     int((self.height - 1) * unit),
                                     fill="gray",
                                     tags="edge")
        self.canvas.create_rectangle(int((self.width - 1.5) * unit),
                                     int((self.height - 1.5) * unit),
                                     int((self.width - 1) * unit),
                                     int((self.height - 1) * unit),
                                     fill="gray",
                                     tags="edge")

        for x in range(unit // 2, (self.width - 2) * unit, unit):
            self.canvas.create_rectangle(x,
                                         0,
                                         x + unit,
                                         unit // 2,
                                         fill="gray",
                                         tags="edge")
            self.canvas.create_rectangle(x,
                                         int((self.height - 1.5) * unit),
                                         x + unit,
                                         int((self.height - 1) * unit),
                                         fill="gray",
                                         tags="edge")
        for y in range(unit // 2, (self.height - 2) * unit, unit):
            self.canvas.create_rectangle(0,
                                         y,
                                         unit // 2,
                                         y + unit,
                                         fill="gray",
                                         tags="edge")
            self.canvas.create_rectangle(int((self.width - 1.5) * unit),
                                         y,
                                         int((self.width - 1) * unit),
                                         y + unit,
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

        self.canvas_map[x][y] = func(x * unit - unit * 3 / 8,
                                     y * unit - unit * 3 / 8,
                                     x * unit + unit * 3 / 8,
                                     y * unit + unit * 3 / 8,
                                     fill=color,
                                     tags=tags)
        return self.canvas_map[x][y]

    def move_chess(self, carvas_obj, x, y):
        old_x = int(
            (self.canvas.coords(carvas_obj)[0] + self.unit / 2) / self.unit)
        old_y = int(
            (self.canvas.coords(carvas_obj)[1] + self.unit / 2) / self.unit)
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
            self.canvas_text[x][y] = self.canvas.create_text(x * unit,
                                                             y * unit,
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
    env.set_chess(1, 1, "oval", "yellow")
    env.set_chess(1, 2, "rectangle", "green")
    env.set_text(1, 3, "X")
    env.mainloop()
    input()
