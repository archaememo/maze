import os
import sys
# import numpy as np
from config import (HEIGHT, WIDTH, N_BLOCK, SAVE_DIR, SAVE_MAP, SAVE_MDL,
                    QUIET, CONFIRM_STOP, DEF_NAME)
from envs.maze import Maze
from envs.maze_gui import MazeGUI
if os.name == "nt":
    GUI = True
else:
    GUI = False

AgentClass = getattr(__import__("agents.dqn_agent", fromlist=["Agent"]),
                     "Agent")

PlayerClass = getattr(__import__("players.nn_player", fromlist=["Player"]),
                      "Player")


def main(file_name=DEF_NAME):
    if GUI:
        maze = MazeGUI(height=HEIGHT, width=WIDTH, n_block=N_BLOCK)
    else:
        maze = Maze(height=HEIGHT, width=WIDTH, n_block=N_BLOCK)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    map_file = SAVE_MAP.format(file_name)
    if not os.path.exists(map_file):
        maze.generate_maze()
        maze.save_map(map_file)
    else:
        maze.load_map(map_file)
    mdl_file = SAVE_MDL.format(file_name)

    player = PlayerClass(AgentClass=AgentClass,
                         maze=maze,
                         quiet=QUIET,
                         confirm_stop=CONFIRM_STOP,
                         mdl_file=mdl_file)

    player.run()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main(DEF_NAME)
