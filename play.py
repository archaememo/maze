from envs.maze import Maze
from envs.maze_gui import MazeGUI
import os
import sys
# =====================================================
QUIET = False
CONFIRM_STOP = True
SAVE_DIR = ".\\store\\"
SAVE_MAP = SAVE_DIR + "{}.map"
SAVE_MDL = SAVE_DIR + "{}.h5"
HEIGHT = 16
WIDTH = 16
N_BLOCK = 64
UNIT = 30
DISPLAY_DELAY = 0
EXPLORE_NUM = 1
GUI = True
# =====================================================


def main(save_file=None):
    if GUI:
        maze = MazeGUI(height=HEIGHT, width=WIDTH, n_block=N_BLOCK, unit=UNIT)
    else:
        maze = Maze(height=HEIGHT, width=WIDTH, n_block=N_BLOCK)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if save_file:
        map_file = SAVE_MAP.format(save_file)
        if not os.path.exists(map_file):
            maze.generate_maze()
            maze.save_map(map_file)
        else:
            maze.load_map(map_file)
        mdl_file = SAVE_MDL.format(save_file)
    else:
        maze.generate_maze()
        map_file = None
        mdl_file = None

    AgentClass = getattr(__import__("agents.dqn_agent", fromlist=["Agent"]),
                         "Agent")

    getattr(__import__("players.nn_player", fromlist=["Player"]),
            "Player")(AgentClass=AgentClass,
                      maze=maze,
                      delay=DISPLAY_DELAY,
                      quiet=QUIET,
                      explorer_num=EXPLORE_NUM,
                      confirm_stop=CONFIRM_STOP,
                      mdl_file=mdl_file)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main(None)
