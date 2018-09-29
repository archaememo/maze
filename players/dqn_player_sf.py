import sys
import numpy as np
import time
sys.path.append('.')
from maps.maze_basic import Maze  # flake8: noqa
from maps.maze_board import MazeBoard  # flake8: noqa
from agents.dqn_agent import DQNAgent  # flake8: noqa


GRID_SIZE = 16
BLOCK_NUM = GRID_SIZE * 3
INPUT_FEATURE = 2
EPISODES = 5000
MAX_STEPS = int(GRID_SIZE * BLOCK_NUM)
LEARN_BATCH = int(GRID_SIZE * GRID_SIZE / 2 )

def play_maze(display_type=0):
    # Iterate the game
    average_steps = 0
    bad_scores = 0
    done = True
    for e in range(EPISODES):
        n_rpos = (1 != e % 3)
        if n_rpos:
            state = env.reset()
        else:
            state = env.reset(r_position=True)
        state = np.reshape(state, [1, INPUT_FEATURE])
        for time_t in range(MAX_STEPS):
            # Decide action
            action, is_predict, raw = agent.act_raw(state)
            old_coords = env.cur_coords.copy()
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, INPUT_FEATURE])
            board.set_me(env.cur_coords)
            if is_predict and display_type:
                display_action(old_coords, action, raw, display_type)
            board.render()
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            if done:
                break

        # stop train if the score is good enough
        if ((average_steps < GRID_SIZE * 2)) and \
           (agent.epsilon < (agent.epsilon_min + 1) / 3):
           break

        if done:
            if time_t < GRID_SIZE * 2:
                agent.enforce_memory(int(time_t), int(time_t), 4)
            else:
                agent.enforce_memory(GRID_SIZE * 2, GRID_SIZE * 2, 4)
                #  agent.enforce_memory(int(time_t - GRID_SIZE), GRID_SIZE, 4)
            # train the agent with the experience of the episode
            # agent.train(batch_size=LEARN_BATCH, mod_epsilon=n_rpos)
            # agent.train_all(mod_epsilon=(1 != e % 3))
        else:
            if not n_rpos:
                agent.forget(time_t)
        
        agent.train(batch_size=LEARN_BATCH, mod_epsilon=False)

        if n_rpos:
            old_step = average_steps
            average_steps = average_steps * 0.9 + time_t * 0.1
            if done:
                bad_scores = max(bad_scores - 1, 0)
                agent.decrease_epsilon()
            else:
                bad_scores += 1
                agent.forget(time_t)
                if bad_scores % 5 == 4:
                    agent.increase_epsilon()
                    bad_scores = 0

        # print the score and break out of the loop
        print("episode: {}/{}, round-steps: {},".format(e, EPISODES, time_t),
              "avg-steps: {}->{}, epsilon: {:.3f}".format(
                  int(old_step), int(average_steps), agent.epsilon))

    show_time()


def show_time():
    print("Show time :-)")
    i = 0
    done = False
    state = env.reset(dis=GRID_SIZE)
    board.set_me(env.cur_coords)
    while not done:
        i = i + 1
        state = np.reshape(state, [1, env.n_features])
        old_coords = env.cur_coords.copy()
        action, _, _ = agent.act_raw(state, True)
        print("step:", i)
        state, _, done = env.step(action, print_track=True)
        board.set_me(env.cur_coords)
        board.set_text(old_coords, "X",color="red")
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
    env = Maze(grid_size=GRID_SIZE, block_num=BLOCK_NUM )
    board = MazeBoard(env.get_r_map(), unit=40)
    agent = DQNAgent(
        INPUT_FEATURE,
        env.n_actions,
        mem_len= MAX_STEPS * 10,
        layer_info=[
            INPUT_FEATURE * GRID_SIZE, GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE,
            GRID_SIZE * env.n_actions
        ])
    board.set_origin(env.cur_coords)
    board.after(100, play_maze, 1)
    board.mainloop()
