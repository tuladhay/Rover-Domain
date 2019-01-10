import numpy as np
import random
cimport cython

cdef extern from "math.h":
    double tanh(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mul(double[:, :] mat, double[:] vec, double[:] out):
    cdef int colIndex, rowIndex
    cdef double sum = 0
    for rowIndex in range(mat.shape[0]):
        sum = 0
        for colIndex in range(mat.shape[1]):
            sum += mat[rowIndex, colIndex] * vec[colIndex]
        out[rowIndex] = sum
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef addInPlace(double[:] vec, double[:] other):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] += other[index]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef addInPlaceMat(double[:,:] mat, double[:,:] other):
    cdef int colIndex, rowIndex
    for rowIndex in range(mat.shape[0]):
        for colIndex in range(mat.shape[1]):
            mat[rowIndex, colIndex] += other[rowIndex, colIndex]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef tanhInPlace(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = tanh(vec[index])
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef reluInPlace(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = vec[index] * (vec[index] > 0)
     
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mutate(double[:] vec, double m, double mr):
    shape = [vec.shape[0]]
    npMutation = np.random.standard_cauchy(shape)
    npMutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:] mutation = npMutation
    addInPlace(vec, mutation)
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mutateMat(double[:, :] mat, double m, double mr):
    shape = [mat.shape[0], mat.shape[1]]
    npMutation = m * np.random.standard_cauchy(shape)
    npMutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:, :] mutation = npMutation
    addInPlaceMat(mat, mutation)
        
cdef class Evo_MLP:
    cdef public double[:, :] in_To_Hidden_Mat
    cdef public double[:] in_To_Hidden_Bias
    cdef public double[:, :] hidden_To_Out_Mat
    cdef public double[:] hidden_To_Out_Bias
    cdef public double[:] hidden # Hidden layer of NN
    cdef public double[:] out # Output layer of NN
    cdef public object np_In_To_Hidden_Mat
    cdef public object np_In_To_Hidden_Bias
    cdef public object np_Hidden_To_Out_Mat
    cdef public object np_Hidden_To_Out_Bias
    cdef public object np_Hidden
    cdef public object np_Out
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
        self.np_In_To_Hidden_Mat = np.random.uniform(-stdev, stdev, (num_units, num_inputs))
        self.np_In_To_Hidden_Bias = np.random.uniform(-stdev, stdev, num_units)
        stdev = (3/ num_units) ** 0.5
        self.np_Hidden_To_Out_Mat = np.random.uniform(-stdev, stdev, (num_outputs, num_units))
        self.np_Hidden_To_Out_Bias = np.random.uniform(-stdev, stdev, num_outputs)
        
        self.np_Hidden = np.zeros(num_units)
        self.np_Out = np.zeros(num_outputs)
        
        self.in_To_Hidden_Mat = self.np_In_To_Hidden_Mat # Input layer to hidden layer
        self.in_To_Hidden_Bias = self.np_In_To_Hidden_Bias # Input layer biasing node
        self.hidden_To_Out_Mat = self.np_Hidden_To_Out_Mat # Hidden layer to output layer
        self.hidden_To_Out_Bias = self.np_Hidden_To_Out_Bias # Hidden layer biasing node
        self.hidden = self.np_Hidden # Hidden layer
        self.out = self.np_Out # Output layer

    cpdef get_action(self, double[:] state):
        mul(self.in_To_Hidden_Mat, state, self.hidden)
        addInPlace(self.hidden, self.in_To_Hidden_Bias)
        reluInPlace(self.hidden)
        mul(self.hidden_To_Out_Mat, self.hidden, self.out)
        addInPlace(self.out, self.hidden_To_Out_Bias)
        tanhInPlace(self.out)
        return self.np_Out

    cpdef mutate(self):
        cdef double m = 1
        cdef double mr = 0.01
        mutateMat(self.in_To_Hidden_Mat, m, mr)
        mutate(self.in_To_Hidden_Bias, m, mr)
        mutateMat(self.hidden_To_Out_Mat, m, mr)
        mutate(self.hidden_To_Out_Bias, m, mr)

        
    cpdef copyFrom(self, other):
        self.num_inputs = other.num_inputs
        self.num_outputs = other.num_outputs
        self.num_units = other.num_units 
        
        cdef double[:, :] new_In_To_Hidden_Mat = other.np_In_To_Hidden_Mat
        self.in_To_Hidden_Mat[:] = new_In_To_Hidden_Mat
        cdef double[:] new_In_To_Hidden_Bias = other.np_In_To_Hidden_Bias
        self.in_To_Hidden_Bias[:] = new_In_To_Hidden_Bias
        cdef double[:, :] new_Hidden_To_Out_Mat = other.np_Hidden_To_Out_Mat
        self.hidden_To_Out_Mat[:] = new_Hidden_To_Out_Mat
        cdef double[:] new_Hidden_To_Out_Bias = other.np_Hidden_To_Out_Bias
        self.hidden_To_Out_Bias[:] = new_Hidden_To_Out_Bias
        

        
        
def init_Ccea(num_inputs, num_outputs, num_units):
    def initCceaGo(data):
        number_agents = data['Number of Agents']

        populationCol = [[Evo_MLP(num_inputs, num_outputs, num_units) for i in range(data['Trains per Episode'])] for j in range(number_agents)]
        data['Agent Populations'] = populationCol
    return initCceaGo
    
def init_Ccea2(num_inputs, num_outputs, num_units):
    def initCceaGo(data):
        number_agents = data['Number of Agents']
        policyCount = data['Number of Policies']
        populationCol = [[Evo_MLP(num_inputs, num_outputs, num_units) for i in range(policyCount)] for j in range(number_agents)]
        data['Agent Populations'] = populationCol
    return initCceaGo
    
def clear_Fitness(data):
    populationCol = data['Agent Populations']
    number_agents = data['Number of Agents']
    
    for agentIndex in range(number_agents):
        for policy in populationCol[agentIndex]:
            policy.fitness = 0
    
def assign_Ccea_Policies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[agentIndex][worldIndex]
    data["Agent Policies"] = policyCol
    
def assign_Ccea_Policies_2(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCount = len(populationCol[0])
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[agentIndex][worldIndex % policyCount]
    data["Agent Policies"] = policyCol
    
def assign_Best_Ccea_Policies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = max(populationCol[agentIndex], key = lambda policy: policy.fitness)
        #policyCol[agentIndex] = populationCol[agentIndex][0]
    data["Agent Policies"] = policyCol

def reward_Ccea_Policies(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(number_agents):
        policyCol[agentIndex].fitness = rewardCol[agentIndex]
 
def reward_Ccea_Policies_2(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(number_agents):
        policyCol[agentIndex].fitness += rewardCol[agentIndex] 
    
cpdef evolve_Ccea_Policies(data):
    cdef int number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    cdef int agentIndex, matchIndex, halfPopLen
    halfPopLen = int(len(populationCol[0])//2)
    for agentIndex in range(number_agents):
        population = populationCol[agentIndex]
        
        # Binary Tournament, replace loser with copy of winner, then mutate copy
        for matchIndex in range(halfPopLen):
            
            if population[2 * matchIndex].fitness > population[2 * matchIndex + 1].fitness:
                population[2 * matchIndex + 1].copyFrom(population[2 * matchIndex])
            else:
                population[2 * matchIndex].copyFrom(population[2 * matchIndex + 1])

            population[2 * matchIndex + 1].mutate()

        random.shuffle(population)
        data['Agent Populations'][agentIndex] = population
        
   
        