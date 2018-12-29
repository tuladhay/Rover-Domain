"""
This module provides a multilayer perception class (Evo_MLP) and Cooperative
    Coevolution Algorithm (CCEA) functionality

Writes:
'Agent Populations' (ArrayLike<ArrayLike<Evo_MLP>>):  A population is a 
    collection of Evo_MLPs, so this is a ArrayLike<Populations>
"Agent Policies" (ArrayLike<Evo_MLP>): Each policy is an Evo_MLP used for acting 
    in the environment.

Reads:
'Number of Agents' (int)
'Number of Policies per Population' (int)
"Number of Inputs" (int)
"Number of Outputs" (int)
"Number of Hidden Units" (int)
'Agent Populations' (ArrayLike<ArrayLike<Evo_MLP>>)
"World Index" (int)
"Agent Policies" (ArrayLike<Evo_MLP>)
"Agent Rewards" (ArrayLike<int>)
"""


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
cdef mutateMat(double[:,:] mat, double m, double mr):
    shape = [mat.shape[0], mat.shape[1]]
    npMutation = m * np.random.standard_cauchy(shape)
    npMutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:,:] mutation = npMutation
    addInPlaceMat(mat, mutation)
        
cdef class Evo_MLP:
    cdef public double[:,:] inToHiddenMat
    cdef public double[:] inToHiddenBias
    cdef public double[:,:] hiddenToOutMat
    cdef public double[:] hiddenToOutBias
    cdef public double[:] hidden
    cdef public double[:] out
    cdef public object npInToHiddenMat
    cdef public object npInToHiddenBias
    cdef public object npHiddenToOutMat
    cdef public object npHiddenToOutBias
    cdef public object npHidden
    cdef public object npOut
    cdef public int inputCount
    cdef public int outputCount
    cdef public int hiddenCount
    cdef public double fitness
    
    def __init__(self, inputCount, outputCount, hiddenCount):
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.hiddenCount = hiddenCount
        self.fitness = 0

        # XAVIER INITIALIZATION
        stdev = (3/ inputCount) ** 0.5
        self.npInToHiddenMat = np.random.uniform(-stdev, stdev, (hiddenCount, inputCount))
        self.npInToHiddenBias = np.random.uniform(-stdev, stdev, hiddenCount)
        stdev = (3/ hiddenCount) ** 0.5
        self.npHiddenToOutMat = np.random.uniform(-stdev, stdev, (outputCount, hiddenCount))
        self.npHiddenToOutBias = np.random.uniform(-stdev, stdev, outputCount)
        
        self.npHidden = np.zeros(hiddenCount)
        self.npOut = np.zeros(outputCount)
        
        self.inToHiddenMat = self.npInToHiddenMat
        self.inToHiddenBias = self.npInToHiddenBias
        self.hiddenToOutMat = self.npHiddenToOutMat
        self.hiddenToOutBias = self.npHiddenToOutBias
        self.hidden = self.npHidden
        self.out = self.npOut

    cpdef get_action(self, double[:] state):
        mul(self.inToHiddenMat, state, self.hidden)
        addInPlace(self.hidden, self.inToHiddenBias)
        reluInPlace(self.hidden)
        mul(self.hiddenToOutMat, self.hidden, self.out)
        addInPlace(self.out, self.hiddenToOutBias)
        tanhInPlace(self.out)
        return self.npOut

    cpdef mutate(self):
        cdef double m = 1
        cdef double mr = 0.01
        mutateMat(self.inToHiddenMat, m, mr)
        mutate(self.inToHiddenBias, m, mr)
        mutateMat(self.hiddenToOutMat, m, mr)
        mutate(self.hiddenToOutBias, m, mr)

        
    cpdef copyFrom(self, other):
        self.inputCount = other.inputCount
        self.outputCount = other.outputCount
        self.hiddenCount = other.hiddenCount 
        
        cdef double[:,:] newInToHiddenMat = other.npInToHiddenMat
        self.inToHiddenMat[:] = newInToHiddenMat
        cdef double[:] newInToHiddenBias = other.npInToHiddenBias
        self.inToHiddenBias[:] = newInToHiddenBias
        cdef double[:,:] newHiddenToOutMat = other.npHiddenToOutMat
        self.hiddenToOutMat[:] = newHiddenToOutMat
        cdef double[:] newHiddenToOutBias = other.npHiddenToOutBias
        self.hiddenToOutBias[:] = newHiddenToOutBias
        
        

def initCcea(inputCount, outputCount, hiddenCount=16):
    "
    Creates
    
    agentCount = data['Number of Agents']
    policyCount = data['Number of Policies per Population']
    inputCount = data["Number of Inputs"]
    outputCount = data["Number of Outputs"]
    hiddenCount = data["Number of Hidden Units"]
    
    populationCol = [[Evo_MLP(inputCount,outputCount,hiddenCount) for i in range(policyCount)] for j in range(agentCount)] 
    data['Agent Populations'] = populationCol
    
def resetFitness(data):
    populationCol = data['Agent Populations']
    agentCount = data['Number of Agents']
    
    for agentIndex in range(agentCount):
        for policy in populationCol[agentIndex]:
            policy.fitness = 0
            
    data['Agent Populations'] = populationCol

    
def assignCceaPolicies(data):
    agentCount = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCount = data['Number of Policies per Population'] 
    policyCol = [None] * agentCount
    for agentIndex in range(agentCount):
        policyCol[agentIndex] = populationCol[agentIndex][worldIndex % policyCount]
    data["Agent Policies"] = policyCol
    
def assignBestCceaPolicies(data):
    agentCount = data['Number of Agents']
    populationCol = data['Agent Populations']
    policyCol = [None] * agentCount
    for agentIndex in range(agentCount):
        policyCol[agentIndex] = max(populationCol[agentIndex], key = lambda policy: policy.fitness)
    data["Agent Policies"] = policyCol

def rewardCceaPolicies(data):
    policyCol = data["Agent Policies"]
    agentCount = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(agentCount):
        policyCol[agentIndex].fitness = rewardCol[agentIndex]
    data["Agent Policies"] = policyCol
    
def addRewardCceaPolicies(data):
    policyCol = data["Agent Policies"]
    agentCount = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(agentCount):
        policyCol[agentIndex].fitness += rewardCol[agentIndex] 
    data["Agent Policies"] = policyCol
    
cpdef evolveCceaPolicies(data): 
    cdef int agentCount = data['Number of Agents']
    populationCol = data['Agent Populations']
    policyCount = data['Number of Policies per Population'] 
    
    cdef int agentIndex, matchIndex, halfPopLen
    halfPopLen = int(policyCount//2)
    for agentIndex in range(agentCount):
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
    data['Agent Populations'] = policyCol
    
   
        