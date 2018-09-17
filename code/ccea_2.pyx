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
cdef tanhInPlace(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = tanh(vec[index])
        

cdef class Mlp:
    cdef double[:,:] inToHiddenMat
    cdef double[:] inToHiddenBias
    cdef double[:,:] hiddenToOutMat
    cdef double[:] hiddenToOutBias
    cdef double[:] hidden
    cdef double[:] out
    
    def process(self, double[:] state):
        mul(self.inToHiddenMat, state, self.hidden)
        addInPlace(self.hidden, self.inToHiddenBias)
        tanhInPlace(self.hidden)
        mul(self.hiddenToOutMat, self.hidden, self.out)
        addInPlace(self.out, self.hiddenToOutBias)
        tanhInPlace(self.out)
        
    def connect(
        self,
        inToHiddenMat, 
        inToHiddenBias,
        hiddenToOutMat,
        hiddenToOutBias,
        hidden,
        out
    ):
        self.out = out
        self.hidden = hidden
        self.hiddenToOutBias = hiddenToOutBias
        self.hiddenToOutMat = hiddenToOutMat
        self.inToHiddenBias = inToHiddenBias
        self.inToHiddenMat = inToHiddenMat
        
        
class Evo_MLP:
    def __init__(self, input_shape, num_outputs, num_units=16):
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.num_units = num_units
        self.fitness = float("-inf")

        self.inToHiddenMat = np.zeros((num_units, input_shape))
        self.inToHiddenBias = np.zeros(num_units)
        self.hiddenToOutMat = np.zeros((num_outputs, num_units))
        self.hiddenToOutBias = np.zeros(num_outputs)
        
        self.hidden = np.zeros(num_units)
        self.out = np.zeros(num_outputs)
        
        self.mlp = Mlp()
        self.mlp.connect(
            self.inToHiddenMat,
            self.inToHiddenBias,
            self.hiddenToOutMat,
            self.hiddenToOutBias,
            self.hidden,
            self.out
        )
        
        
    def get_action(self, state):
        self.mlp.process(state)
        return self.out


    def mutate(self):
        m = 10
        mr = 0.01
        weightCol = [self.inToHiddenMat, self.inToHiddenBias, self.hiddenToOutMat, self.hiddenToOutBias]
        for weight in weightCol:
            mutation = np.random.normal(0, m, list(weight.shape))
            mutation *= np.random.uniform(size = list(weight.shape)) < mr
            weight += mutation

        
    def copyFrom(self, other):
        self.inToHiddenMat[:] = other.inToHiddenMat
        self.inToHiddenBias[:] = other.inToHiddenBias
        self.hiddenToOutMat[:] = other.hiddenToOutMat
        self.hiddenToOutBias[:] = other.hiddenToOutBias
        
        self.mlp.connect(
            self.inToHiddenMat,
            self.inToHiddenBias,
            self.hiddenToOutMat,
            self.hiddenToOutBias,
            self.hidden,
            self.out
        )
        
        
def initCcea(input_shape, num_outputs, num_units=16):
    def initCceaGo(data):
        number_agents = data['Number of Agents']
        populationCol = [[Evo_MLP(input_shape,num_outputs,num_units) for i in range(data['Trains per Episode'])] for j in range(number_agents)] 
        data['Agent Populations'] = populationCol
    return initCceaGo
    
def assignCceaPolicies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[agentIndex][worldIndex]
    data["Agent Policies"] = policyCol
    
def assignBestCceaPolicies(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = max(populationCol[agentIndex], key = lambda policy: policy.fitness)
        #policyCol[agentIndex] = populationCol[agentIndex][0]
    data["Agent Policies"] = policyCol

def rewardCceaPolicies(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(number_agents):
        policyCol[agentIndex].fitness = rewardCol[agentIndex]
    
def evolveCceaPolicies(data): 
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    for agentIndex in range(number_agents):
        population = populationCol[agentIndex]

        # Binary Tournament, replace loser with copy of winner, then mutate copy
        for matchIndex in range(len(population)//2):
            if population[2 * matchIndex].fitness > population[2 * matchIndex + 1].fitness:
                population[2 * matchIndex + 1].copyFrom(population[2 * matchIndex])
                newPolicy = population[2 * matchIndex + 1]
            else:
                population[2 * matchIndex].copyFrom(population[2 * matchIndex + 1])
                newPolicy = population[2 * matchIndex + 1]

            newPolicy.mutate()

        random.shuffle(population)
        data['Agent Populations'][agentIndex] = population
        