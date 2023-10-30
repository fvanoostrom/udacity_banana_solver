import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [64,64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): cd p1_naviRandom seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()

        input_size = state_size
        for size in hidden_layers:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            #add rectified linear unit
            self.layers.append(nn.ReLU())
        #add output layer
        self.layers.append(nn.Linear(size, action_size))

        # if hidden_layers is None:
        #     self.fc1 = nn.Linear(state_size, action_size)
        # else:
        #     self.fc1 = nn.Linear(state_size, hidden_layers[0])
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        data = state
        for layer in self.layers:
            data = layer(data)
        return data

    # def forward(self, state):
    #     """Build a network that maps state -> action values."""
    #     x = F.relu(self.fc1(state))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)
