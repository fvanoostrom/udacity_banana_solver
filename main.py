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

env.print_env_info()

# base_agent = BaseAgent(state_size=37, action_size=4, seed=0)
# score = env.run_episode(base_agent, 20)

# print("Score: {}".format(score))

configuration = {
                "max_episodes" : 2000 ,
                "max_time" : 500, 
                "eps_start" : 1.0,
                "eps_end" : 0.01,
                "eps_decay" : 0.997,
                "target_score" : 14.0,
                "agent" : {
                        "type" : "double_dqn",
                        "buffer_size" : int(1e6),  # replay buffer size
                        "batch_size" : 128,         # minibatch size
                        "gamma" : 0.99,            # discount factor
                        "tau" : 1e-3,              # for soft update of target parameters
                        "lr" : 5e-4,               # learning rate 
                        "update_every" : 4,        # how often to update the network
                        "network" :{"hidden_layers": [256, 256]} #amount and size of hidden layers
                }
}

agents ={"base": BaseAgent, "random": BaseAgent, "dqn": DQNAgent, "double_dqn": DoubleDQNAgent}

start_date = datetime.now()
name = configuration["agent"]["type"] + "_" + start_date.strftime("%Y%m%d_%H%M%S")
model_path = 'output/model_' + name + '.pt'

agent = agents[configuration["agent"]["type"]](state_size=37, action_size=4, seed=0, agent_configuration = configuration["agent"])
scores = env.train(agent, configuration["max_episodes"], configuration["max_time"], 
                   configuration["eps_start"], configuration["eps_end"], configuration["eps_decay"],
                   configuration["target_score"],)
agent.save(path=model_path)

end_date = datetime.now()
duration = end_date - start_date

cur_result = {"type":configuration["agent"]["type"], "date": start_date.strftime("%Y-%m-%d %H:%M:%S"), "episodes" : len(scores),
              "final_score" :  sum(scores[-100:])/len(scores[-100:]), "duration" : str(duration),
              "scores": scores, "configuration": configuration, "model_path" : model_path}

# save the result of the current run
with open("output/results_"+ name + ".json", 'w') as f:
    json.dump(cur_result, f, indent=2) 

# open all previous results
if os.path.isfile("output/results.json"):
    with open("output/results.json", 'r') as f:
        results = json.load(f)
# if it does not exist 
else:
    results = []

#save results with the current result appended
results.append(cur_result)
with open("output/results.json", 'w') as f:
    json.dump(results, f, indent=2)


# plot the results and compare them with other results
def calculate_moving_average(numbers, window_size):
    return [round(sum(numbers[max(0,i-window_size):i]) / 
                  min(i,window_size),
                  2)
            for i in range(1, len(numbers)+1)]

plt.plot(calculate_moving_average(scores,100), color='red', label='current result')
cmap = plt.cm.get_cmap('hsv', len(results)+2)

#filter the results on the 5 best final score (highest average of last 100 episodes) 
results_displayed = sorted(results, key=lambda r: r['final_score'], reverse=True)[:5]
#plot these results by taking all the scores and calculating the moving average.
for i, result in enumerate(results_displayed):
    plt.plot(calculate_moving_average(result['scores'],100)[:len(scores)], color=cmap(i+1), alpha=0.5, linestyle='--', label=result['type'] + " " + result['date'])
plt.legend()
plt.ylabel("Score")
plt.xlabel("Episode #")
# plt.show()
plt.savefig("output/" + name + ".png")




# close the environment
env.close()
