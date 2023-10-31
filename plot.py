import json
import os
import matplotlib.pyplot as plt

# plot the results and compare them with other results
def calculate_moving_average(numbers, window_size):
    return [round(sum(numbers[max(0,i-window_size):i]) / 
                  min(i,window_size),
                  2)
            for i in range(1, len(numbers)+1)]

# open all previous results
if os.path.isfile("output/results.json"):
    with open("output/results.json", 'r') as f:
        results = json.load(f)

# plot the most recent results
number_of_results = 10
column_to_filter = 'date'
reverse = True
results_displayed = sorted(results, key=lambda r: r[column_to_filter], reverse=reverse)[:number_of_results]

#plot these results by taking all the scores and calculating the moving average.
plt.rcParams["figure.figsize"] = (20,8)

#have a different color for each result
cmap = plt.cm.get_cmap('hsv', len(results_displayed)+1)
for i, result in enumerate(results_displayed):
    label = result['type'] + '_' + result['date'] if result.get('alias') is None else result['alias']
    plt.plot(calculate_moving_average(result['scores'],100), color=cmap(i+1), alpha=1.0, linestyle='-', label=label)
plt.legend()
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.show()