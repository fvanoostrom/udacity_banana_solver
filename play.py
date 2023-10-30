from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from rl_environment import RLEnvironment

from base_agent import BaseAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent

import random
from datetime import datetime
import json
import os
import re

env = RLEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

configuration = {
                "max_episodes" : 2000,
                "max_time" : 300, 
                "eps_start" : 1.0,
                "eps_end" : 0.01,
                "eps_decay" : 0.990,
                "target_score" : 100.0,
                "agent" : {
                        "type" : "double_dqn",
                        "buffer_size" : int(1e5),  # replay buffer size
                        "batch_size" : 128,         # minibatch size
                        "gamma" : 0.99,            # discount factor
                        "tau" : 1e-3,              # for soft update of target parameters
                        "lr" : 5e-4,               # learning rate 
                        "update_every" : 4,        # how often to update the network
                        "network" :{"hidden_layers": [64,64]} #amount and size of hidden layers
                }
}

agents ={"base": BaseAgent, "random": BaseAgent, "dqn": DQNAgent, "double_dqn": DoubleDQNAgent}

agent = agents[configuration["agent"]["type"]](state_size=37, action_size=4, seed=0, agent_configuration = configuration["agent"])
agent.load('output/model_double_dqn_20231030_124559.pt')
score = env.run_episode(agent)
print(score)

# close the environment
env.close()
