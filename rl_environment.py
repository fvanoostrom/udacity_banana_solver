from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from base_agent import BaseAgent
from dqn_agent import DQNAgent

class RLEnvironment(UnityEnvironment):
    """Takes care of the Reinforcement Learning Environment"""

    def __init__(self, file_name):
        """Initialize an Agent object.
        
        Params
        ======
            file_name (string) : name of the executable file to run
        """
        super().__init__(file_name=file_name)
        self.brain_name = self.brain_names[0]
        self.brain = self.brains[self.brain_name]
        self.initiated = False

    def reset(self, train_mode=True, config=None, lesson=None):
        self.env_info = super().reset(train_mode, config, lesson)
        self.brain_info = self.env_info[self.brain_name]
        self.initiated = True
        return self.env_info

    def print_env_info(self):
        if not self.initiated:
            self.reset()
        # number of agents in the environment
        print('Number of agents:', len(self.brain_info.agents))

        # number of actions
        action_size = self.brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space 
        state = self.brain_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

    def run_episode(self, agent, train_mode = False, max_t = 2000, eps = 0.0):
        # reset the environment
        self.reset(train_mode=train_mode) 
        state = self.brain_info.vector_observations[0]            # get the current state

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            self.brain_info = self.step(action)[self.brain_name]        # send the action to the environment
            next_state = self.brain_info.vector_observations[0]   # get the next state
            reward = self.brain_info.rewards[0]                   # get the reward
            done = self.brain_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        return score
    
        
    def train(self, agent, max_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_score= 5.0 ):
        """Deep Q-Learning.
        
        Params
        ======
            agent (agent): agent to use
            max_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, max_episodes+1):
            score = self.run_episode(agent, True, max_t, eps)

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=target_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                agent.save()
                break
        return scores


if __name__ == "__main__":

    env = RLEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

    env.print_env_info()

    agent = BaseAgent(state_size=37, action_size=4, seed=0)
    score = env.run_episode(agent, 20)
    print("Score: {}".format(score))
    
    agent = DQNAgent(state_size=37, action_size=4, seed=0)
    scores = env.train(agent, 10, 10, target_score = 5.0)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    #close the environment
    env.close()