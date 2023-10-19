from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from base_agent import BaseAgent
from dqn_agent import DQNAgent
from rl_environment import RLEnvironment

env = RLEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

env.print_env_info()

# base_agent = BaseAgent(state_size=37, action_size=4, seed=0)
# score = env.run_episode(base_agent, 20)

# print("Score: {}".format(score))

dqn_agent = DQNAgent(state_size=37, action_size=4, seed=0)
scores = env.train(dqn_agent, 1000, 200, target_score=5.0)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()

# close the environment
env.close()
