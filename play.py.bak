from envs.maze import Maze
# =====================================================
QUIET = False
CONFIRM_STOP = True
FACTOR = 1
SAVE_DIR = ".\\stored_maps\\{:.0f}.maze"
HEIGHT = 16
WIDTH = 16
N_BLOCK = 64
UNIT = 30
DISPLAY_DELAY = 0
EXPLORE_NUM = 1
# =====================================================


def main():

    maze = Maze(height=HEIGHT, width=WIDTH, n_block=N_BLOCK, unit=UNIT)
    maze.generate_maze()
    # maze.save_map(SAVE_DIR.format(time.time()))
    # maze.load_map(SAVE_DIR.format(1539350556))

    AgentClass = getattr(__import__("agents.dqn_agent", fromlist=["Agent"]),
                         "Agent")

    getattr(__import__("players.nn_player", fromlist=["Player"]),
            "Player")(AgentClass=AgentClass,
                      maze=maze,
                      delay=DISPLAY_DELAY,
                      quiet=QUIET,
                      explorer_num=EXPLORE_NUM,
                      confirm_stop=CONFIRM_STOP,
                      factor=FACTOR)


if __name__ == "__main__":
    main()
