# Player configure
# =============================================
CONFIRM_STOP = True
DISPLAY_DELAY = 0.1
EXPLORE_NUM = 1
MAX_EPISODE = 100000
INC_EPISODE_THRESHOLD = 2
VERIFY_EPISODE = 32
VERIFY_THRESHOLD = 3
VERIFY_TIME = 2
ACCURACY_THRESHOLD = 0.8

# ================================================

# Maze configure
# =============================================
QUIET = False
SAVE_DIR = ".\\store\\"
DEF_NAME = "mymaze"
SAVE_MAP = SAVE_DIR + "{}.map"
SAVE_MDL = SAVE_DIR + "{}.h5"
HEIGHT = 16
WIDTH = 16
N_BLOCK = 64
BLOCK = -1  # reward & mark of block
GOAL = 1  # reward & mark of goal
EMPTY = 0  # reward & mark of empty
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# GUI
UNIT = 30
# ================================================

# Agent configure
# =================================================
SAVE_MDL = SAVE_DIR + "{}.h5"
EPSILON_INIT = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.01
GAMMA = 0.8
MEM_LEN = 1000
PARAM_FILE = "{}.para"

# nn
LAYER_INFO = [128, 128]
EVALUATE_EPISODE = 8
BATCH_SIZE = 64
PARAM_FILE = "{}.para"
# =================================================
