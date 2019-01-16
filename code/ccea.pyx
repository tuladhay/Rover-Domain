import numpy as np
from parameters import parameters as p
import random

cdef class ccea:
    cdef public double fitness = 0.0
    cdef public double mut_prob = p.data["Mutation Rate"]
    cdef public double epsilon = p.data["Epsilon"]
    cdef public int n_populations = p.data["Number of Agents"]
    cdef public int population_size  = p.data["Population Size"]*2
    n_inputs = p.data["Number of Inputs"]
    n_outputs = p.data["Number of Outputs"]
    n_nodes = p.data["Number of Nodes"]  # Number of nodes in hidden layer
    cdef public int policy_size = (n_inputs + 1)*n_nodes + (n_nodes + 1)*n_outputs  # Number of weights for NN

    populations = np.zeros(n_populations, population_size)
    cdef double[:] pops = populations

    fitness = np.zeros(n_populations, population_size)
    cdef double[:] fit_vec = fitness

    cpdef __init__(self):

        # Initialize a random population of NN weights
        for pop_index in range(self.n_populations):
            for policy_index in range(self.population_size):
                policy = [0.0 for i in range(self.policy_size)]
                for w in range(self.policy_size):
                    policy[w] = random.uniform(0, 1)
                self.pops[pop_index, policy_index] = policy[:]

    cpdef reset_populations(self):
        for pop_index in range(self.n_populations):
            for policy_index in range(self.population_size):
                policy = [0.0 for i in range(self.policy_size)]
                for w in range(self.policy_size):
                    policy[w] = random.uniform(0, 1)
                self.pops[pop_index, policy_index] = policy[:]

    cpdef mutate(self, half_length):
        for pop_index in range(self.n_populations):
            policy_index = half_length
            while policy_index < self.population_size:
                rvar = random.uniform(0, 1)
                if rvar <= self.mut_prob:
                    target = random.randint(0, (self.policy_size - 1))
                    self.pops[pop_index][policy_index][target] = random.uniform(0, 1)
                policy_index += 1

    cpdef create_new_pop(self):
        cdef int count = 0
        cdef int half_pop_length = self.population_size/2
        for pop_index in range(self.n_populations):
            policy_index = half_pop_length
            while policy_index < self.population_size:
                for w in range(self.policy_size):
                    self.pops[pop_index][policy_index][w] = self.pops[pop_index][policy_index][count]
                policy_index += 1
                count += 1

        self.mutate(half_pop_length)

    cpdef epsilon_greedy_select(self):
        cdef int half_pop_length = self.population_size/2
        for pop_id in range(self.n_populations):
            policy_id = half_pop_length
            while policy_id < self.population_size:
                rvar = random.uniform(0, 1)
                if rvar <= self.epsilon:  # Choose best policy
                    for k in range(self.policy_size):
                        self.pops[pop_id][policy_id][k] = self.pops[pop_id][0][k]  # Best policy
                else:
                    parent = random.randint(0, self.population_size - 1)  # Choose a random parent
                    for k in range(self.policy_size):
                        self.pops[pop_id][policy_id][k] = self.pops[pop_id][parent][k]  # Random policy
                policy_id += 1

    cpdef down_select(self):
        # Reorder populations in terms of fintess (greatest to least)
        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                k = j + 1
                while k < self.population_size:
                    if self.fit_vec[pop_id][j] < self.fit_vec[pop_id][k]:
                        self.fit_vec[pop_id][j], self.fit_vec[pop_id][k] = self.fit_vec[pop_id][k], self.fit_vec[pop_id][j]
                        self.pops[pop_id][j], self.pops[pop_id][k] = self.pops[pop_id][k], self.pops[pop_id][j]
                    k += 1

        self.epsilon_greedy_select()