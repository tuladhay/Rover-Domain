import numpy as np
import random
cimport cython

cdef extern from "math.h":
    double tanh(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mul(double[:, :] mat, double[:] vec, double[:] out):
    cdef int column_id, row_id
    cdef double temp_sum = 0
    for row_id in range(mat.shape[0]):
        temp_sum = 0
        for column_id in range(mat.shape[1]):
            temp_sum += mat[row_id, column_id] * vec[column_id]
        out[row_id] = temp_sum

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef add_in_place(double[:] vec, double[:] other):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] += other[index]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef add_in_place_mat(double[:,:] mat, double[:,:] other):
    cdef int column_id, row_id
    for row_id in range(mat.shape[0]):
        for column_id in range(mat.shape[1]):
            mat[row_id, column_id] += other[row_id, column_id]
        
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
cdef mutate_mat(double[:, :] mat, double m, double mr):
    shape = [mat.shape[0], mat.shape[1]]
    np_mutation = m * np.random.standard_cauchy(shape)
    np_mutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:, :] mutation = np_mutation
    add_in_place_mat(mat, mutation)
        
cdef class Evo_MLP:
    cdef public double[:, :] in_to_hidden_mat
    cdef public double[:] in_to_hidden_bias
    cdef public double[:, :] hidden_to_out_mat
    cdef public double[:] hidden_to_out_bias
    cdef public double[:] hidden # Hidden layer of NN
    cdef public double[:] out # Output layer of NN
    cdef public object np_in_to_hidden_mat
    cdef public object np_in_to_hidden_bias
    cdef public object np_hidden_to_out_mat
    cdef public object np_hidden_to_out_bias
    cdef public object np_hidden
    cdef public object np_out
    cdef public int num_inputs
    cdef public int num_outputs
    cdef public int num_units
    cdef public double fitness

    
    def __init__(self, num_inputs, num_outputs, num_units):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_units = num_units # Size of hidden layer
        self.fitness = 0

        # XAVIER INITIALIZATION
        stdev = (3/ num_inputs) ** 0.5
        self.np_in_to_hidden_mat = np.random.uniform(-stdev, stdev, (num_units, num_inputs))
        self.np_in_to_hidden_bias = np.random.uniform(-stdev, stdev, num_units)
        stdev = (3/ num_units) ** 0.5
        self.np_hidden_to_out_mat = np.random.uniform(-stdev, stdev, (num_outputs, num_units))
        self.np_hidden_to_out_bias = np.random.uniform(-stdev, stdev, num_outputs)
        
        self.np_hidden = np.zeros(num_units)
        self.np_out = np.zeros(num_outputs)
        
        self.in_to_hidden_mat = self.np_in_to_hidden_mat # Input layer to hidden layer
        self.in_to_hidden_bias = self.np_in_to_hidden_bias # Input layer biasing node
        self.hidden_to_out_mat = self.np_hidden_to_out_mat # Hidden layer to output layer
        self.hidden_to_out_bias = self.np_hidden_to_out_bias # Hidden layer biasing node
        self.hidden = self.np_hidden # Hidden layer
        self.out = self.np_out # Output layer


    cpdef get_action(self, double[:] state):
        mul(self.in_to_hidden_mat, state, self.hidden)
        add_in_place(self.hidden, self.in_to_hidden_bias)
        relu_in_place(self.hidden)
        mul(self.hidden_to_out_mat, self.hidden, self.out)
        add_in_place(self.out, self.hidden_to_out_bias)
        tanh_in_place(self.out)
        return self.np_out


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
        
        cdef double[:, :] new_In_To_Hidden_Mat = other.np_in_to_hidden_mat
        self.in_to_hidden_mat[:] = new_In_To_Hidden_Mat
        cdef double[:] new_in_to_hidden_bias = other.np_in_to_hidden_bias
        self.in_to_hidden_bias[:] = new_in_to_hidden_bias
        cdef double[:, :] new_hidden_to_out_mat = other.np_hidden_to_out_mat
        self.hidden_to_out_mat[:] = new_hidden_to_out_mat
        cdef double[:] new_hidden_to_out_bias = other.np_hidden_to_out_bias
        self.hidden_to_out_bias[:] = new_hidden_to_out_bias
        

        
def init_ccea(num_inputs, num_outputs, num_units):
    def initCceaGo(data):
        number_agents = data['Number of Agents']

        population = [[Evo_MLP(num_inputs, num_outputs, num_units) for i in range(data['Trains per Episode'])] for j in range(number_agents)]
        data['Agent Populations'] = population
    return initCceaGo

    
def init_ccea2(num_inputs, num_outputs, num_units):
    def initCceaGo(data):
        number_agents = data['Number of Agents']
        policy_count = data['Number of Policies']
        population = [[Evo_MLP(num_inputs, num_outputs, num_units) for i in range(policy_count)] for j in range(number_agents)]
        data['Agent Populations'] = population
    return initCceaGo

    
def clear_fitness(data):
    population = data['Agent Populations']
    number_agents = data['Number of Agents']
    
    for agent_id in range(number_agents):
        for policy in population[agent_id]:
            policy.fitness = 0

    
def assign_ccea_policies(data):
    number_agents = data['Number of Agents']
    population = data['Agent Populations']
    world_index = data["World Index"]
    policyCol = [None] * number_agents
    for agent_id in range(number_agents):
        policyCol[agent_id] = population[agent_id][world_index]
    data["Agent Policies"] = policyCol

    
def assign_ccea_policies_2(data):
    number_agents = data['Number of Agents']
    population = data['Agent Populations']
    world_index = data["World Index"]
    policy_count = len(population[0])
    policyCol = [None] * number_agents
    for agent_id in range(number_agents):
        policyCol[agent_id] = population[agent_id][world_index % policy_count]
    data["Agent Policies"] = policyCol

    
def assign_best_ccea_policies(data):
    number_agents = data['Number of Agents']
    population = data['Agent Populations']
    policyCol = [None] * number_agents
    for agent_id in range(number_agents):
        policyCol[agent_id] = max(population[agent_id], key = lambda policy: policy.fitness)
        #policyCol[agent_id] = population[agent_id][0]
    data["Agent Policies"] = policyCol


def reward_ccea_policies(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agent_id in range(number_agents):
        policyCol[agent_id].fitness = rewardCol[agent_id]

 
def reward_ccea_policies_2(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agent_id in range(number_agents):
        policyCol[agent_id].fitness += rewardCol[agent_id]

    
cpdef evolve_ccea_policies(data):
    cdef int number_agents = data['Number of Agents']
    population = data['Agent Populations']
    cdef int agent_id, matchIndex, halfPopLen
    halfPopLen = int(len(population[0])//2)
    for agent_id in range(number_agents):
        population = population[agent_id]
        
        # Binary Tournament, replace loser with copy of winner, then mutate copy
        for matchIndex in range(halfPopLen):
            
            if population[2 * matchIndex].fitness > population[2 * matchIndex + 1].fitness:
                population[2 * matchIndex + 1].copy_from(population[2 * matchIndex])
            else:
                population[2 * matchIndex].copy_from(population[2 * matchIndex + 1])

            population[2 * matchIndex + 1].mutate()

        random.shuffle(population)
        data['Agent Populations'][agent_id] = population
        
   
        