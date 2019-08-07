from players.abstract_player import AbstractPlayer
from envs.maze import Maze
from config import (VERIFY_EPISODE, BATCH_SIZE, MEM_LEN)
import multiprocessing as mp
from queue import Full, Empty
import os
import random
import time
import sys

EXPLORE_NUM = 1
CFG_T_INIT = 0
PARAM_T_SCORE_AND_WEIGHT = 0


class Player(AbstractPlayer):
    def __init__(self, **kwargs):
        super(Player, self).__init__(**kwargs)

    @staticmethod
    def trainer_proc(cfg_pipe, mem_que, param_que):
        (_, (AgentClass, state_size, action_size, mdl_file)) = cfg_pipe.recv()
        agent = AgentClass(state_size=state_size,
                           action_size=action_size,
                           mdl_file=mdl_file)
        for _ in range(BATCH_SIZE):
            agent.remember(mem_que.get(True))
        episode = 0
        print("trainer enter work status")
        while True:
            agent.train()
            episode += 1
            if (episode % VERIFY_EPISODE == VERIFY_EPISODE - 1):
                agent.evaluate()
                agent.save_model()
                try:
                    param_que.put_nowait(
                        (PARAM_T_SCORE_AND_WEIGHT, (agent.score,
                                                    agent.get_weights())))
                    # print("[%s], model score: %s" %
                    #       (sys._getframe().f_code.co_name, agent.score))
                except Full:
                    pass
            for _ in range(BATCH_SIZE):
                try:
                    agent.remember(mem_que.get_nowait())
                except Empty:
                    break

    @staticmethod
    def explorer_proc(cfg_pipe, mem_que, param_que):
        (_, (AgentClass, state_size, action_size, epsilon,
             maze_map_file)) = cfg_pipe.recv()
        agent = AgentClass(state_size=state_size,
                           action_size=action_size,
                           mdl_file=None)
        env = Maze().load_map(maze_map_file)
        agent.modify_epsilon(epsilon)
        print("explore enter work status")
        episode = 0
        while True:
            episode += 1
            env.reset(random.random() < epsilon)
            _, cache = Player.explore(env, agent, True)
            for r in cache:
                mem_que.put(r)
                # try:
                #     mem_que.put_nowait(r)
                # except Full:
                #     time.sleep(0.5)
                    # print("[%s]: sleep for a while" %
                    #       (sys._getframe().f_code.co_name))
                    # break
            if episode % VERIFY_EPISODE == VERIFY_EPISODE - 1:
                try:
                    (para_type, (score, weights)) = param_que.get_nowait()
                    agent.set_weights(weights)
                    # debug
                    # print("[%s]: update weight, epsilon=%f" %
                    #     (sys._getframe().f_code.co_name, epsilon))
                    if param_que.empty():
                        param_que.put_nowait((para_type, (score, weights)))
                except (Empty, Full):
                    pass

    def start_processes(self):
        self.mem_que = mp.Queue(MEM_LEN)
        self.param_que = mp.Queue(10)

        self.train_pipe, train_pipe = mp.Pipe()
        self.works = [
            mp.Process(target=Player.trainer_proc,
                       args=(train_pipe, self.mem_que, self.param_que))
        ]

        self.explore_pipes = []
        for _ in range(EXPLORE_NUM):
            explore_pipe = mp.Pipe()
            self.explore_pipes.append(explore_pipe[0])
            self.works.append(
                mp.Process(target=Player.explorer_proc,
                           args=(explore_pipe[1], self.mem_que,
                                 self.param_que)))
        for work in self.works:
            work.start()

        self.train_pipe.send(
            (CFG_T_INIT,
             (self.AgentClass, self.maze.observation_space.shape[0],
              self.maze.action_space.n, self.mdl_file)))

        self.t_map = "{}.tmp".format(int(time.time() * 1000))

        self.maze.save_map(self.t_map)

        for p in self.explore_pipes:
            p.send((CFG_T_INIT,
                    (self.AgentClass, self.maze.observation_space.shape[0],
                     self.maze.action_space.n, 0.5, self.t_map)))

    def stop_processes(self):
        [work.terminate() for work in self.works]
        [work.join() for work in self.works]
        os.remove(self.t_map)

    def run(self):
        self.init_run_var()
        self.start_processes()
        episode = 0
        while True:
            episode += 1
            # _, cache = Player.explore(self.maze, self.agent, True)
            if self.verify():
                break
            if episode % VERIFY_EPISODE == VERIFY_EPISODE - 1:
                try:
                    _, (score, weights) = self.param_que.get()
                    self.agent.set_weights(weights)
                    print("[%s], update model with loss/accuary: %s" %
                          (sys._getframe().f_code.co_name, score))
                except (Full, Empty):
                    continue
                except Exception:
                    self.stop_processes()
                    return
        self.exit()
