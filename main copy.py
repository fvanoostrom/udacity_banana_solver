from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import copy

from rl_environment import RLEnvironment
from base_agent import BaseAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent

agents ={"base": BaseAgent, "random": BaseAgent, "dqn": DQNAgent, "double_dqn": DoubleDQNAgent}

base_configuration = {
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


param_sweeps = {"eps_end": [0.05, 0.02, 0.01, 0.00], "eps_decay" : [0.990, 0.995, 0.997, 0.999]}

param_sweeps = {"hidden_layers" : [[12],[16],[24],[32],[48],[64],[96],[128],[12, 12],[16, 16],[24, 24],[32, 32],[48, 48],[64, 64],[96, 96],[128, 128],[16, 16, 16],[24, 24, 24],[32, 32, 32],[48, 48, 48],[64, 64, 64],[96, 96, 96],[128, 128, 128]]}
param_sweeps = {"gamma" : [0.98,0.99,0.995], "tau": [1e-2,1e-3,1e-4], "lr": [5e-3,5e-4,5e-5]}
param_sweeps = {"type" : ["double_dqn","double_dqn","double_dqn","dqn","dqn","dqn"]}

param_sweeps = {"hidden_layers" : [[32, 32],[48, 48],[64, 64]]}


def create_configurations(base_configuration, param_sweeps):
    configurations = [base_configuration]
    for key in param_sweeps.keys():
        new_configurations = []
        for value in param_sweeps[key]:
            for config in configurations:
                new_config = copy.deepcopy(config)
                # new_config[key] = value
                new_config['agent']['network'][key] = value
                new_configurations.append(new_config)
        configurations = new_configurations

    for configuration in configurations:
        configuration['alias'] = configuration["agent"]["type"]
        for key in param_sweeps.keys():
            configuration['alias'] += " " + key + ":" + str(configuration['agent']['network'][key])

    return configurations

configurations = create_configurations(base_configuration, param_sweeps)

env = RLEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

env.print_env_info()

for configuration in configurations:
    start_date = datetime.now()
    name = configuration["agent"]["type"] + "_" + start_date.strftime("%Y%m%d_%H%M%S")
    model_path = 'output/model_' + name + '.pt'
    # create an agent based on the configuration
    agent = agents[configuration["agent"]["type"]](state_size=37, action_size=4, seed=0, agent_configuration = configuration["agent"])
    # run the training
    scores = env.train(agent, configuration["max_episodes"], configuration["max_time"], 
                    configuration["eps_start"], configuration["eps_end"], configuration["eps_decay"],
                    configuration["target_score"],)
    agent.save(path=model_path)

    end_date = datetime.now()
    duration = end_date - start_date

    cur_result = {"name": name, "alias": configuration["alias"],"type":configuration["agent"]["type"],
                  "date": start_date.strftime("%Y-%m-%d %H:%M:%S"), "episodes" : len(scores),
                  "final_score" :  sum(scores[-100:])/len(scores[-100:]), "duration" : str(duration),
                  "scores": scores, "configuration": configuration, "model_path" : model_path}

    # save the result of the current run
    with open("output/results_"+ name + ".json", 'w') as f:
        json.dump(cur_result, f, indent=2) 

    # open all previous results
    if os.path.isfile("output/results.json"):
        with open("output/results.json", 'r') as f:
            results = json.load(f)
    # if it does not exist create an empty array
    else:
        results = []

    # save results with the current result appended
    results.append(cur_result)
    with open("output/results.json", 'w') as f:
        json.dump(results, f, indent=2)


# plot the results and compare them with other results
def calculate_moving_average(numbers, window_size):
    return [round(sum(numbers[max(0,i-window_size):i]) / 
                  min(i,window_size),
                  2)
            for i in range(1, len(numbers)+1)]

# plt.plot(calculate_moving_average(scores,100), color='red', label='current result')
#filter the results on the 5 best final score (highest average of last 100 episodes) 
# results_displayed = sorted(results, key=lambda r: r['final_score'], reverse=True)[:5]
#filter the most recent results 
results_displayed = sorted(results, key=lambda r: r['date'], reverse=True)[:len(configurations)]


#plot these results by taking all the scores and calculating the moving average.
plt.rcParams["figure.figsize"] = (20,8)
#have a different color for each result
cmap = plt.cm.get_cmap('hsv', len(results_displayed)+2)
for i, result in enumerate(results_displayed):
    label = result['type'] + '_' + result['date'] if result.get('alias') is None else result['alias']
    plt.plot(calculate_moving_average(result['scores'],100), color=cmap(i+1), alpha=1.0, linestyle='-', label=label)
plt.legend()
plt.ylabel("Score")
plt.xlabel("Episode #")
# plt.show()
plt.savefig("output/" + name + ".png", bbox_inches='tight', dpi=600)




# close the environment
env.close()
