import numpy as np
from parameters import Parameters as p
import random
import pyximport; pyximport.install() # For cython(pyx) code

cdef class Ccea:
    cdef public double mut_prob
    cdef public double epsilon
    cdef public int n_populations
    cdef public int population_size
    cdef public int policy_size
    cdef public double[:, :, :] pops
    cdef public double[:, :] fitness
    cdef public int[:, :] team_selection

    def __init__(self):
        self.mut_prob = p.data["Mutation Rate"]
        self.epsilon = p.data["Epsilon"]
        self.n_populations = p.data["Number of Agents"]  # One population for each rover
        self.population_size  = p.data["Population Size"]*2  # Number of policies in each pop
        n_inputs = p.data["Number of Inputs"]
        n_outputs = p.data["Number of Outputs"]
        n_nodes = p.data["Number of Nodes"]  # Number of nodes in hidden layer
        self.policy_size = (n_inputs + 1)*n_nodes + (n_nodes + 1)*n_outputs  # Number of weights for NN

        self.pops = np.zeros((self.n_populations, self.population_size, self.policy_size))
        self.fitness = np.zeros((self.n_populations, self.population_size))
        self.team_selection = np.zeros((self.n_populations, self.population_size), dtype = np.int32)

        cdef int pop_index, policy_index, w
        # Initialize a random population of NN weights
        for pop_index in range(self.n_populations):
            for policy_index in range(self.population_size):
                for w in range(self.policy_size):
                    self.pops[pop_index, policy_index, w] = random.uniform(0, 1)
                self.team_selection[pop_index, policy_index] = -1

    cpdef select_policy_teams(self):
        cdef int pop_id, policy_id, j, k, rpol
        for pop_id in range(self.n_populations):
            for policy_id in range(self.population_size):
                self.team_selection[pop_id, policy_id] = -1

        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                rpol = random.randint(0, (self.population_size - 1))
                k = 0
                while k < j:  # Make sure unique number is chosen
                    if rpol == self.team_selection[pop_id, k]:
                        rpol = random.randint(0, (self.population_size - 1))
                        k = -1
                    k += 1
                self.team_selection[pop_id, j] = rpol  # Assign policy to team

    cpdef reset_populations(self):
        cdef int pop_index, policy_index, w
        for pop_index in range(self.n_populations):
            for policy_index in range(self.population_size):
                for w in range(self.policy_size):
                    self.pops[pop_index, policy_index, w] = random.uniform(0, 1)

    cpdef mutate(self, half_length):
        cdef int pop_index, policy_index, target
        cdef double rvar
        for pop_index in range(self.n_populations):
            policy_index = half_length
            while policy_index < self.population_size:
                rvar = random.uniform(0, 1)
                if rvar <= self.mut_prob:
                    target = random.randint(0, (self.policy_size - 1))
                    self.pops[pop_index, policy_index, target] = random.uniform(0, 1)
                policy_index += 1

    cpdef create_new_pop(self):
        cdef int count, policy_index, pop_index, w
        cdef int half_pop_length = self.population_size/2
        for pop_index in range(self.n_populations):
            policy_index = half_pop_length
            count = 0
            while policy_index < self.population_size:
                for w in range(self.policy_size):
                    self.pops[pop_index, policy_index, w] = self.pops[pop_index, count, w]
                policy_index += 1
                count += 1

        self.mutate(half_pop_length)

    cpdef epsilon_greedy_select(self):
        cdef int pop_id, policy_id, k, parent
        cdef double rvar
        cdef int half_pop_length = self.population_size/2
        for pop_id in range(self.n_populations):
            policy_id = half_pop_length
            while policy_id < self.population_size:
                rvar = random.uniform(0, 1)
                if rvar <= self.epsilon:  # Choose best policy
                    for k in range(self.policy_size):
                        self.pops[pop_id, policy_id, k] = self.pops[pop_id, 0, k]  # Best policy
                else:
                    parent = random.randint(0, self.population_size - 1)  # Choose a random parent
                    for k in range(self.policy_size):
                        self.pops[pop_id, policy_id, k] = self.pops[pop_id, parent, k]  # Random policy
                policy_id += 1

    cpdef down_select(self):
        cdef int pop_id, j, k
        # Reorder populations in terms of fintess (greatest to least)
        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                k = j + 1
                while k < self.population_size:
                    if self.fitness[pop_id, j] < self.fitness[pop_id, k]:
                        self.fitness[pop_id, j], self.fitness[pop_id, k] = self.fitness[pop_id, k], self.fitness[pop_id, j]
                        self.pops[pop_id, j], self.pops[pop_id, k] = self.pops[pop_id, k], self.pops[pop_id, j]
                    k += 1

        self.epsilon_greedy_select()