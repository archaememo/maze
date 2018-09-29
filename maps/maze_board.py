import numpy as np
# import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # pixels
BLOCK = -1  # mark of block
GOAL = 100  # reward & mark of goal
EMPTY = 0  # reward of empty


class MazeBoard(tk.Tk, object):
    def __init__(self, r_map, direction=False, unit=UNIT):
        super(MazeBoard, self).__init__()
        # top left coords (x, y) distance between me and goal
        self.grid_size = len(r_map[0])
        self.unit = unit
        self.title('maze')
        self.geometry('{0}x{1}'.format(self.grid_size * unit,
                                       self.grid_size * unit))

        # map of maze
        self.r_map = r_map

        # canvas map of maze
        self.c_map = [[0 for i in range(self.grid_size)]
                      for j in range(self.grid_size)]

        self.direction = direction

        self.me = 0

        self.origin_coords = np.array([-1, -1])

        # draw the maze
        self._init_window()

    def _init_window(self):
        grid_size = self.grid_size
        unit = self.unit
        info = self.r_map

        self.canvas = tk.Canvas(
            self, bg='white', height=grid_size * unit, width=grid_size * unit)

        cv = self.canvas

        # create grids
        for c in range(0, (grid_size + 1) * unit, unit):
            x0, y0, x1, y1 = c, 0, c, grid_size * unit
            cv.create_line(x0, y0, x1, y1)
        for r in range(0, (grid_size + 1) * unit, unit):
            x0, y0, x1, y1 = 0, r, grid_size * unit, r
            cv.create_line(x0, y0, x1, y1)

        # draw the block and goal
        for i in range(grid_size):
            for j in range(grid_size):
                if BLOCK == info[i][j]:
                    self.c_map[i][j] = cv.create_rectangle(
                        i * unit + unit / 8,
                        j * unit + unit / 8,
                        i * unit + unit * 7 / 8,
                        j * unit + unit * 7 / 8,
                        fill="black",
                        tags="block")
                elif GOAL == info[i][j]:
                    self.c_map[i][j] = cv.create_oval(
                        i * unit + unit / 8,
                        j * unit + unit / 8,
                        i * unit + unit * 7 / 8,
                        j * unit + unit * 7 / 8,
                        fill="yellow",
                        tags="goal")
                elif self.direction:
                    self.c_map[i][j] = cv.create_text(
                        i * unit + unit / 2,
                        j * unit + unit / 2,
                        fill="gray",
                        text="?")
        self.canvas.pack()

    def set_me(self, cur_coords):
        unit = self.unit
        if 0 != self.me:
            old = np.array([(int(self.canvas.coords(self.me)[0])),
                            (int(self.canvas.coords(self.me)[1]))]) // unit
            self.canvas.move(self.me, unit * (cur_coords[0] - old[0]),
                             unit * (cur_coords[1] - old[1]))
        else:
            self.me = self.canvas.create_rectangle(
                cur_coords[0] * unit + unit / 8,
                cur_coords[1] * unit + unit / 8,
                cur_coords[0] * unit + unit * 7 / 8,
                cur_coords[1] * unit + unit * 7 / 8,
                fill='green',
                tags='me')

    def set_origin(self, cur_coords):
        self.origin_coords = cur_coords.copy()

    def set_text(self, cur_coords, txt, color='gray'):
        unit = self.unit
        if not (self.origin_coords - cur_coords).any():
            fill_color = 'purple'
        else:
            fill_color = color

        if 0 != self.c_map[cur_coords[0]][cur_coords[1]]:
            self.canvas.itemconfig(
                self.c_map[cur_coords[0]][cur_coords[1]],
                text=txt,
                fill=fill_color,
                tags="text")
        else:
            self.c_map[cur_coords[0]][cur_coords[1]] = self.canvas.create_text(
                cur_coords[0] * unit + unit / 2,
                cur_coords[1] * unit + unit / 2,
                fill=fill_color,
                text=txt,
                tags='text')

    def clear_text(self):
        for w in range(self.grid_size):
            for h in range(self.grid_size):
                if 0 != self.c_map[w][h]:
                    if self.canvas.gettags(self.c_map[w][h])[0] == 'text':
                        self.canvas.itemconfig(self.c_map[w][h], text="")

    def render(self):
        self.update()


if __name__ == "__main__":
    test_map = [[0, 0, 0, BLOCK], [0, BLOCK, 0, 0], [BLOCK, 0, 0, 0],
                [0, 0, 0, GOAL]]
    env = MazeBoard(test_map)
    env.set_me(np.array([0, 0]))
    env.mainloop()
