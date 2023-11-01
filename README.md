[//]: # (Image References)

[image1]: results/trained_agent.gif "Trained Agent"

# Banana Agent
## Project Details


Within this project we will develop an agent that will navigate and pick up bananas within a square arena. This project is part of the Udacity Deep learning Program and is an adapted version of the Unity  [ML-Agents repository](https://github.com/Unity-Technologies/ml-agents).

In the image belowathe final trained agent within the environment is shown. The agent receives 37 inputs ("the state space is 37"), such as the velocity and ray-based perception of objects and has four possible actions (forward, backward, turn left, turn right). Whenever it touches a banana it will pick it up. When it picks up a yellow banana it will receive a reward of +1. Picking up a blue banana results in a reward of -1. It is thus the goal to pick up as many yellow bananas as possible while avoiding picking up blue bananas. 

![Trained Agent][image1]

## Getting Started
This is a python project. In order to run it you will need follow these steps:
1. install Visual Studio Code or another python editor. In case you are running Visual Studio Code install an older version of the 'Python' extension (v2021.9.1246542782) since the Python 3.6 is no longer supported in more recent versions.
1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
1. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
1. Clone this repository (if you haven't already!)

1. install the requirements in requirements.txt by typing the following command in the terminal
    ```
    pip install -r requirements.txt
    ```

1. Download the banana environment within the same folder as you have put this project
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


1. Change the `file_name` parameter within main.py and Navigation.ipynb to match the location of the Unity environment that you downloaded.

    - **Mac**: `"path/to/Banana.app"`
    - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
    - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
    - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
    - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
    - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
    - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`

    For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
    ```
    env = UnityEnvironment(file_name="Banana.app")
    ```

You are now ready to run the agent!

## Instructions
In order to train the agent you can either run the jupyter notebook Navigation.ipynb or the python file main.py. Since python 3.6 is no longer supported Navigation.ipynb will only run in the Udacity environment, not run in visual studio code. You can run main.py will using F5 (debug) or CTRL + F5 (run without debugging). 

A Unity Environment application should start up within Visual Studio Code. The Udacity environment cannot display the application. The python script connects to this environment and sends commands (turn left, right, forward, backwards) to the environment. You will see a sped up version of the game in order to make the training faster. The python file will train an agent. Depending on the machine it should take about 10-30 minutes. Afterwards the trained model of the agent is saved within 'model.pt' file. Moreover, the results, model and matplotlib-graph will be saved to the output folder.
