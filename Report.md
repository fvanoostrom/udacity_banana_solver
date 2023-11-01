[//]: # (Image References)

[image1]: results/eps.png "Results eps"
[image2]: results/buffersize_batchsize.png "Results buffer and batch size"
[image3]: results/hidden_layers.png "Results hidden layers"
[image4]: results/gamma_tau_lr.jpg "Results tau gamma lr parameter sweeping"
[image5]: results/dqn_vs_double_dqn.png "Results dqn versus double dqn"

Learning Algorithm

*The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.*

Plot of Rewards

*A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.*

# Architecture
The architecture used is based on the architecture of the LunarLander exercise. This is a Q-learning algorithm that uses deep learning to learn the right values. For this exercise both a regular and a double dqn network were used. In the latter one, an additional network is created that is used during the learning phase. This should reduce the overestimation of a value.

The solution is structured in this way:
- main.py
- base_agent: a base class of an agent which performs random actions
- double_dqn_agent: an agent with a double dqn architecture, based on the 
- dqn_agent: an agent with a dqn architecture

- dnn_model: a class with the 'brain' of the agent, 

# Trying out different configurations of hyperparameters
There are a lot of parameters we can tweak in order to reach a good performing model as soon as possible. In order to learn which parameters work well we will try out different configurations. 
Different configurations were tested. The goal was to reach an average score over the last 100 episodes above 14.0. If the goal was not reached within 200 episodes the training was stopped.

## Eps
The paramter EPS is about how often the algorithm should take a decision based on it's current knowledge and how often it should do a random action. In the beginning, it should always take random actions because it has no knowledge yet. The amount of randomness should decay over time. A eps_decay value of 0.99 means that after one episode the eps has a value of 0.99 and the algorithm should take 99% of the time a random action. After 100 episodes a value of 0.99^100 = 36.7% is reached. Which means after 100 episodes the algorithm more often makes a decision based on it's still limited knowledge than based on randomness. If instead a value of 0.999 is used it takes 1000 episodes to reach this value. 

In order to still learn after some time a treshold can be set to have at least some randomness. This parameter is the 'eps_end'. A value of 0.01 means in 1% of the time the algorithm will still take a random action. 

The following values were used for eps_decay and eps_end
Eps_decay
- 0.99
- 0.995
- 0.997
- 0.999

eps_end:
- 0.0
- 0.01
- 0.02
- 0.05

The results in the image below make clear that a too high value for the eps_decay have a big influence on the performance. 
The parameter eps_end seemed to have very little effect, since the target score was often reached before the eps value would drop below the eps_end value.

![Results eps][image1]

## Buffer size and batch size
Another set of hyperparameters are the buffer size and batch size. These determine how many of the states are saved in memory, and how many of those states are used for each learn phase. By default these values are set to 100.000 states and a batch size of 64. Other values used were:

- buffer_size: 1000, 10000, 100000
- batch_size: 64, 128, 256

Having a smaller buffer size seems to negatively impact the performance. Having a buffer size of just 1000 means the algorithm can only learn from just over 3 episodes. The batch_size seems to less have an impact. A batch size of 128 seems to be optimal.

![Results buffer and batch size][image2]

## Hidden Layers
Next, different amount and sizes of hidden layers were tried out. One, two and three layers of hidden layers were tried out with each a size of 12, 16, 24, 32, 48, 64, 96 and 128. Most results were very close to each other, with one clear exception: the configuration with three hidden layers of 128 took 1843 episodes.


![Results hidden layers][image3]

Since the plot is difficult to read 

-  [32, 32, 32] :  387
-  [32, 32] :  403
-  [48, 48] :  419
-  [64, 64, 64] :  424
-  [64] :  440
-  [24, 24] :  441
-  [24, 24, 24] :  446
-  [96] :  449
-  [64, 64] :  466
-  [16, 16, 16] :  467
-  [32] :  471
-  [24] :  472
-  [48, 48, 48] :  481
-  [96, 96, 96] :  486
-  [16, 16] :  507
-  [48] :  513
-  [12] :  537
-  [12, 12] :  540
-  [128] :  550
-  [16] :  558
-  [96, 96] :  560
-  [128, 128] :  634
-  [128, 128, 128] :  1843

The results show that having more than 1 hidden layer does not seem to benefit the performance a lot, althoug. 

## Gamma, Tau and Learning Rate
Next the gamma, tau and lr (learning rate) parameters were explored. For each of these parameters 3 values were tried, which leads to 3 x 3 x 3 = 27 different combinations.

These were the different values used for the parameters
- gamma : 0.98, 0.99, 0.995
- tau: 0.01, 0.001, 0.0001
- lr: 0.005, 0.0005, 0.0005

In the figure below you can see progress of moving average of each of these configurations. Some of these converge very fast to a good performing result, while others in the first few hundred episodes perform well, but still take quite long to reach an average above 14.0. 
![Results tau gamma lr parameter sweeping][image4]

As the image above can be difficult to read, these are the 10 best performing combinations:
- gamma: 0.99 tau: 0.001 lr: 5e-05 :  367 episodes
- gamma: 0.99 tau: 0.001 lr: 0.0005 :  403 episodes
- gamma: 0.995 tau: 0.001 lr: 5e-05 :  422 episodes
- gamma: 0.98 tau: 0.01 lr:5e-05 :  459 episodes
- gamma: 0.98 tau: 0.001 lr: 5e-05 :  465 episodes
- gamma: 0.98 tau: 0.001 lr: 0.0005 :  475 episodes
- gamma: 0.995 tau: 0.001 lr: 0.0005 :  491 episodes
- gamma: 0.98 tau: 0.01 lr: 0.0005 :  538 episodes
- gamma: 0.99 tau: 0.01 lr: 5e-05 :  588 episodes
- gamma: 0.99 tau: 0.01 lr: 0.0005 :  593 episodes   
   
A gamma of 0.99, tau of 0.001 and learning rate of 5e-05 seem to be the best performing values, and the combination of these three was the best performing in which it took just 367 episodes to reach an average score of 14.0. These values are quite similar to the default values previously used. Only the Learning rate is set to a 5e-04 by default in stead of 5e-05.

- type: double_dqn
- eps_decay: 0.99
- eps_end: 0.01 or 0.02
- hidden_layers: [32, 32]
- gamma: 0.99
- tau: 0.001
- lr: 5e-05
 
## Double DQN versus DQN
Finally Double DQN is compared to DQN. Each algorithm was ran three times with the optimal parameters found previously. When looking at the graph the results of each of algorithm are quite consistent, with the double dqn algorithm clearly outperforming the algorithm with just one model. Furthermore, none of the double dqn runs come close to previous results. It took 529, 650 and 659 episodes to achieve an average score of over 14.0. While previously a result of 367 episodes was achieved. Simply combining the best hyperparamters might not lead to the best result.
![double dqn compared to dqn][image5]



# Ideas for Future Work 
When looking at the trained agent is becomes clear that the agent prefers chasing small but more immediate rewards in the form of a single close banana instead of going to a further spot where multiple yellow bananas are in close proximity. This might be due to insufficient training, a gamma set too small or another cause. Further research is required to improve on this.

On a technical side, there is currently no prioritization applied when running or storing the memory with all the states. Prioritized experience replay could improve the agent. 