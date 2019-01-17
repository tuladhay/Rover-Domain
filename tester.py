from parameters import Parameters as p
from code.ccea import Ccea
from code.neural_network import  NeuralNetwork
import random

"""
This is just a temporary file to test the CCEA and NN to make sure they function
as expected.
"""


# Test CCEA architecture
cc = Ccea()

generations = 1
for g in range(generations):

    cc.create_new_pop()
    # for i in range(cc.n_populations):
    #     for j in range(cc.population_size):
    #         for k in range(cc.policy_size):
    #             print(cc.pops[i, j, k], end = ' ')
    #         print('\n')

    cc.select_policy_teams()
    # for i in range(cc.n_populations):
    #     for j in range(cc.population_size):
    #         print(cc.team_selection[i, j], end = ' ')
    #     print('\n')

    for i in range(cc.n_populations):
        for j in range(cc.population_size):
            cc.fitness[i, j] = random.uniform(0, 10)

    cc.down_select()
    cc.reset_populations()

# Test Neural Network Architecture
nn = NeuralNetwork()
n_rovers = p.data["Number of Agents"]
input_vec = [1 for i in range(nn.n_inputs)]
weights = [1 for i in range(nn.n_weights)]
for i in range(n_rovers):
    nn.get_inputs(input_vec, i)
    nn.get_weights(weights, i)
    output = nn.get_outputs(i)
    print('Input Layer: ', nn.in_layer[i, 0])
    print('Hidden Layer: ', nn.hid_layer[i, 0])
    print('Outputs: ', output[0], output[1])
nn.reset_nn()
