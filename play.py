import json
import os
from rl_environment import RLEnvironment
from base_agent import BaseAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent


env = RLEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

result_path = 'output/results_double_dqn_20231030_124559.json'

if os.path.isfile(result_path):
        with open(result_path, 'r') as f:
                result = json.load(f)

configuration = result['configuration']
model_path = result['model_path']


agents ={"base": BaseAgent, "random": BaseAgent, "dqn": DQNAgent, "double_dqn": DoubleDQNAgent}

agent = agents[configuration["agent"]["type"]](state_size=37, action_size=4, seed=0, agent_configuration = configuration["agent"])
agent.load(model_path)
score = env.run_episode(agent)
print(score)

# close the environment
env.close()
