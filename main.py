from typing import Dict
import sys
from agent import agent
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from lux.game import Game

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simple.dqn import DQN, ReplayMemory

if __name__ == "__main__":
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    # env = gym.make("CartPole-v1")
    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    n_actions = 8
    # n_actions = env.action_space.n
    n_observations = 8
    # state, info = env.reset()
    # n_observations = len(state)
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    state = random.sample(range(1, 50), 8)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)


    step = 0


    class Observation(Dict[str, any]):
        def __init__(self, player=0) -> None:
            self.player = player
            # self.updates = []
            # self.step = 0


    observation = Observation()
    observation["updates"] = []
    observation["step"] = 0
    player_id = 0
    last_map_state = None
    while True:
        inputs = read_input()
        observation["updates"].append(inputs)

        if step == 0:
            player_id = int(observation["updates"][0])
            observation.player = player_id
        if inputs == "D_DONE":
            global game_state

            ### Do not edit ###
            if observation["step"] == 0:
                game_state = Game()
                game_state._initialize(observation["updates"])
                game_state._update(observation["updates"][2:])
                game_state.id = observation.player
            else:
                game_state._update(observation["updates"])

            possible_actions = []

            actions = agent(observation, None)
            observation["updates"] = []
            step += 1
            observation["step"] = step
            print(",".join(actions))
            print("D_FINISH")