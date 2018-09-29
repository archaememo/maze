import sys
import numpy as np
import time
sys.path.append('.')
from maps.maze_basic import Maze  # flake8: noqa
from maps.maze_board import MazeBoard  # flake8: noqa
from agents.ql_agent import QLAgent  # flake8: noqa

GRID_SIZE = 16
BLOCK_NUM = GRID_SIZE * 5
INPUT_FEATURE = GRID_SIZE * GRID_SIZE
EPISODES = 2000
MAX_STEPS = int(GRID_SIZE * BLOCK_NUM )




get_state = lambda x: int(x[1] * GRID_SIZE + x[0])

def play_maze(display_type=0):
    # Iterate the game
    average_steps = 0
    bad_scores = 0
    done = True
    for e in range(EPISODES):
        if 1 == e % 3:
            _ = env.reset(r_position=True)
        else:
            _ = env.reset()
        state = get_state(env.cur_coords)
        for time_t in range(MAX_STEPS):
            # Decide action
            action, is_predict, raw = agent.act_raw(state)
            old_coords = env.cur_coords.copy()
            _, reward, done = env.step(action)
            next_state =get_state(env.cur_coords)
            board.set_me(env.cur_coords)
            if is_predict and display_type:
                display_action(old_coords, action, raw, display_type)
            board.render()

            # Remember the previous state, action, reward, and done
            agent.remember(state, int(action), reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            # done becomes True when the game ends
            # ex) The agent drops the pole

            if done:
                break

        # stop train if the score is good enough
        if ((average_steps < GRID_SIZE * 2)) and \
           (agent.epsilon < (agent.epsilon_min + 1) / 2):
            break

        if not done:
            bad_scores += 1
            if bad_scores > 10:
                agent.increase_epsilon()
                bad_scores = 0
        else:
            bad_scores = max(bad_scores - 1, 0)
            if (1 != e % 3):
                agent.decrease_epsilon()

        if (1 != e % 3):
            old_step = average_steps
            average_steps = average_steps * 0.9 + time_t * 0.1

        # print the score and break out of the loop
        print("episode: {}/{}, round-steps: {},".format(e, EPISODES, time_t),
              "avg-steps: {}->{}, epsilon: {:.3f}".format(
                  int(old_step), int(average_steps), agent.epsilon))

    show_time()


def show_time():
    print("Show time :-)")
    i = 0
    done = False
    _ = env.reset(dis=GRID_SIZE)
    board.set_me(env.cur_coords)
    board.render()
    while not done:
        i = i + 1
        state = get_state(env.cur_coords)
        old_coords = env.cur_coords.copy()
        action, _, _ = agent.act_raw(state, True)
        print("step:", i)
        _, _, done = env.step(action, print_track=True)
        board.set_me(env.cur_coords)
        board.set_text(old_coords, "X","red")
        board.render()
        time.sleep(0.5)


def display_action(coords, action, raw_action, display_type=1):
    if 2 == display_type:
        board.set_text(
            coords, "U:{:+.2f}\nD:{:+.2f}\nR:{:+.2f}\nL:{:+.2f}".format(
                raw_action[0], raw_action[1], raw_action[2], raw_action[3]))
    else:
        if 0 == action:
            board.set_text(coords, "U")
        if 1 == action:
            board.set_text(coords, "D")
        if 2 == action:
            board.set_text(coords, "R")
        if 3 == action:
            board.set_text(coords, "L")


if __name__ == "__main__":
    env = Maze(grid_size=GRID_SIZE, block_num=BLOCK_NUM)
    board = MazeBoard(env.get_r_map(), unit=40)
    agent = QLAgent(
        INPUT_FEATURE,
        env.n_actions)
    board.after(100, play_maze, 1)
    board.mainloop()
