import numpy as np
import random
cimport cython

cdef extern from "math.h":
    double tanh(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mul(double[:, :] mat, double[:] vec, double[:] out):
    cdef int col_index, row_index
    cdef double sum = 0
    for row_index in range(mat.shape[0]):
        sum = 0
        for col_index in range(mat.shape[1]):
            sum += mat[row_index, col_index] * vec[col_index]
        out[row_index] = sum

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef add_in_place(double[:] vec, double[:] other):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] += other[index]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef add_in_place_mat(double[:,:] mat, double[:,:] other):
    cdef int col_index, row_index
    for row_index in range(mat.shape[0]):
        for col_index in range(mat.shape[1]):
            mat[row_index, col_index] += other[row_index, col_index]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef tanh_in_place(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = tanh(vec[index])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef relu_in_place(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = vec[index] * (vec[index] > 0)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef mutate(double[:] vec, double m, double mr):
    shape = [vec.shape[0]]
    np_mutation = np.random.standard_cauchy(shape)
    np_mutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:] mutation = np_mutation
    add_in_place(vec, mutation)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef mutate_mat(double[:,:] mat, double m, double mr):
    shape = [mat.shape[0], mat.shape[1]]
    np_mutation = m * np.random.standard_cauchy(shape)
    np_mutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:,:] mutation = np_mutation
    add_in_place_mat(mat, mutation)

cdef class Evo_MLP:
    cdef public double[:,:] in_to_hidden_mat
    cdef public double[:] in_to_hidden_bias
    cdef public double[:,:] hidden_to_out_mat
    cdef public double[:] hidden_to_out_bias
    cdef public double[:] hidden
    cdef public double[:] out
    cdef public object np_in_to_hidden_mat
    cdef public object np_in_to_hidden_bias
    cdef public object np_hidden_to_out_mat
    cdef public object np_hidden_to_out_bias
    cdef public object np_hidden_layer
    cdef public object np_output_layer
    cdef public int num_inputs
    cdef public int num_outputs
    cdef public int num_units # Number of nodes in hidden layer
    cdef public double fitness

    def __init__(self, n_inputs, n_outputs, n_units):
        self.num_inputs = n_inputs
        self.num_outputs = n_outputs
        self.num_units = n_units
        self.fitness = 0

        # XAVIER INITIALIZATION
        stdev = (3/ n_inputs) ** 0.5
        self.np_in_to_hidden_mat = np.random.uniform(-stdev, stdev, (n_units, n_inputs))
        self.np_in_to_hidden_bias = np.random.uniform(-stdev, stdev, n_units)
        stdev = (3/ n_units) ** 0.5
        self.np_hidden_to_out_mat = np.random.uniform(-stdev, stdev, (n_outputs, n_units))
        self.np_hidden_to_out_bias = np.random.uniform(-stdev, stdev, n_outputs)

        self.np_hidden_layer = np.zeros(n_units)
        self.np_output_layer = np.zeros(n_outputs)

        self.in_to_hidden_mat = self.np_in_to_hidden_mat
        self.in_to_hidden_bias = self.np_in_to_hidden_bias
        self.hidden_to_out_mat = self.np_hidden_to_out_mat
        self.hidden_to_out_bias = self.np_hidden_to_out_bias
        self.hidden = self.np_hidden_layer
        self.out = self.np_output_layer

    cpdef get_action(self, double[:] state):
        mul(self.in_to_hidden_mat, state, self.hidden)
        add_in_place(self.hidden, self.in_to_hidden_bias)
        relu_in_place(self.hidden)
        mul(self.hidden_to_out_mat, self.hidden, self.out)
        add_in_place(self.out, self.hidden_to_out_bias)
        tanh_in_place(self.out)
        return self.np_output_layer

    cpdef mutate(self):
        cdef double m = 1
        cdef double mr = 0.01
        mutate_mat(self.in_to_hidden_mat, m, mr)
        mutate(self.in_to_hidden_bias, m, mr)
        mutate_mat(self.hidden_to_out_mat, m, mr)
        mutate(self.hidden_to_out_bias, m, mr)


    cpdef copy_from(self, other):
        self.num_inputs = other.num_inputs
        self.num_outputs = other.num_outputs
        self.num_units = other.num_units

        cdef double[:,:] new_in_to_hidden_mat = other.np_in_to_hidden_mat
        self.in_to_hidden_mat[:] = new_in_to_hidden_mat
        cdef double[:] new_in_to_hidden_bias = other.np_in_to_hidden_bias
        self.in_to_hidden_bias[:] = new_in_to_hidden_bias
        cdef double[:,:] new_hidden_to_out_mat = other.np_hidden_to_out_mat
        self.hidden_to_out_mat[:] = new_hidden_to_out_mat
        cdef double[:] new_hidden_to_out_bias = other.np_hidden_to_out_bias
        self.hidden_to_out_bias[:] = new_hidden_to_out_bias




def init_ccea(num_inputs, num_outputs, num_units):
    def init_ccea_go(data):
        number_agents = data['Number of Agents']

        agent_populations = [[Evo_MLP(num_inputs, num_outputs, num_units) for i in range(data['Trains per Episode'])] for j in range(number_agents)]
        data['Agent Populations'] = agent_populations
    return init_ccea_go

def init_ccea2(num_inputs, num_outputs, num_units):
    def init_ccea_go(data):
        number_agents = data['Number of Agents']
        policy_count = data['Number of Policies']
        agent_populations = [[Evo_MLP(num_inputs, num_outputs, num_units) for i in range(policy_count)] for j in range(number_agents)]
        data['Agent Populations'] = agent_populations
    return init_ccea_go

def clear_fitness(data):
    agent_populations = data['Agent Populations']
    number_agents = data['Number of Agents']

    for agent_id in range(number_agents):
        for policy in agent_populations[agent_id]:
            policy.fitness = 0

def assign_ccea_policies(data):
    number_agents = data['Number of Agents']
    agent_populations = data['Agent Populations']
    world_index = data["World Index"]
    agent_policies = [None] * number_agents
    for agent_id in range(number_agents):
        agent_policies[agent_id] = agent_populations[agent_id][world_index]
    data["Agent Policies"] = agent_policies

def assign_ccea_policies2(data):
    number_agents = data['Number of Agents']
    agent_populations = data['Agent Populations']
    world_index = data["World Index"]
    policy_count = len(agent_populations[0])
    agent_policies = [None] * number_agents
    for agent_id in range(number_agents):
        agent_policies[agent_id] = agent_populations[agent_id][world_index % policy_count]
    data["Agent Policies"] = agent_policies

def assign_best_ccea_policies(data):
    number_agents = data['Number of Agents']
    agent_populations = data['Agent Populations']
    agent_policies = [None] * number_agents
    for agent_id in range(number_agents):
        agent_policies[agent_id] = max(agent_populations[agent_id], key = lambda policy: policy.fitness)
        #agent_policies[agent_id] = agent_populations[agent_id][0]
    data["Agent Policies"] = agent_policies

def reward_ccea_policies(data):
    agent_policies = data["Agent Policies"]
    number_agents = data['Number of Agents']
    agent_rewards = data["Agent Rewards"]
    for agent_id in range(number_agents):
        agent_policies[agent_id].fitness = agent_rewards[agent_id]

def reward_ccea_policies2(data):
    agent_policies = data["Agent Policies"]
    number_agents = data['Number of Agents']
    agent_rewards = data["Agent Rewards"]
    for agent_id in range(number_agents):
        agent_policies[agent_id].fitness += agent_rewards[agent_id]

cpdef evolve_ccea_policies(data):
    cdef int number_agents = data['Number of Agents']
    agent_populations = data['Agent Populations']
    cdef int agent_id, match_index, half_pop_len
    half_pop_len = int(len(agent_populations[0])//2)
    for agent_id in range(number_agents):
        population = agent_populations[agent_id]

        # Binary Tournament, replace loser with copy of winner, then mutate copy
        for match_index in range(half_pop_len):

            if population[2 * match_index].fitness > population[2 * match_index + 1].fitness:
                population[2 * match_index + 1].copy_from(population[2 * match_index])
            else:
                population[2 * match_index].copy_from(population[2 * match_index + 1])

            population[2 * match_index + 1].mutate()

        random.shuffle(population)
        data['Agent Populations'][agent_id] = population