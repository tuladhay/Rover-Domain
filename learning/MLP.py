"""
Creates a simple MLP model to be trained via evolution
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """
    Neural network structure to control the agents.
    """
    def __init__(self, inputs, hidden, outputs):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(inputs, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, outputs)

    def forward(self, inputs):
        """
        Calculates the output of the network
        :param inputs: input to the neural network
        :return: output of action-space dimension
        """
        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def mutate(self, ):
        """
        Mutates the network in-place
        Mutate 10 Percent of the weights by relative 10 percent
        :return: None
        """
        # Mutate each layer
        for layer in [self.linear1, self.linear2, self.linear3]:
            fc_params = layer.state_dict()
            # Mutate weights
            shape = fc_params['weight'].shape
            num_changed = int(shape[0]*shape[1]*0.1)
            if num_changed == 0:
                # For small networks, increase it to select 1 node with 50% chance
                if random.random() < 0.5:
                    num_changed = 1
            inds = random.sample(list(range(shape[0]*shape[1])), num_changed)
            for ind in inds:
                x = ind%shape[0]
                y = ind//shape[0]
                print("Changing weight", x, y)
                delta = fc_params['weight'][x, y] * 0.1
                if random.random() < 0.5:
                    fc_params['weight'][x, y] += delta
                else:
                    fc_params['weight'][x, y] -= delta
            # Mutate Bias
            shape = fc_params['bias'].shape
            num_changed = int(shape[0]*0.1)
            if num_changed == 0:
                # For small networks, increase it to select 1 node with 50% chance
                if random.random() < 0.5:
                    num_changed = 1
            inds = random.sample(list(range(shape[0])), num_changed)
            for ind in inds:
                print("Changing bias", ind)
                delta = fc_params['bias'][ind] * 0.1
                if random.random() < 0.5:
                    fc_params['bias'][ind] += delta
                else:
                    fc_params['bias'][ind] -= delta
            # Reload the modified state dict
            layer.load_state_dict(fc_params)


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    model = Policy(2, 2, 2)
    print(model.state_dict())
    model.mutate()
    print(model.state_dict())
