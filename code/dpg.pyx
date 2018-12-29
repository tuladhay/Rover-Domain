cimport cython
import numpy as np
import random
from libc.math cimport tanh
 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef matMul(double[:, :] mat, double[:] vec, double[:] out):
    cdef int colIndex, rowIndex
    cdef double sum = 0
    for rowIndex in range(mat.shape[0]):
        sum = 0
        for colIndex in range(mat.shape[1]):
            sum += mat[rowIndex, colIndex] * vec[colIndex]
        out[rowIndex] = sum
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef add(double[:] vecA, double[:] vecB, double[:] out):
    cdef int index
    for index in range(out.shape[0]):
        out[index] = vecA[index] + vecB[index]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.        
cdef sub(double[:] vecA, double[:] vecB, double[:] out):
    cdef int index
    for index in range(out.shape[0]):
        out[index] = vecA[index] - vecB[index]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef addMat(double[:,:] matA, double[:,:] matB, double[:,:] out):
    cdef int colIndex, rowIndex
    for rowIndex in range(out.shape[0]):
        for colIndex in range(out.shape[1]):
            out[rowIndex, colIndex] = matA[rowIndex, colIndex] + matB[rowIndex, colIndex]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef tanhv(double[:] vec, double[:] out):
    cdef int index
    for index in range(out.shape[0]):
        out[index] = tanh(vec[index])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef relu(double[:] vec, double[:] out):
    cdef int index
    for index in range(out.shape[0]):
        out[index] = vec[index] * (vec[index] > 0)
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.        
cdef transpose(double[:, :] mat, double[:, :] out):
    cdef int colIndex, rowIndex
    for rowIndex in range(out.shape[0]):
        for colIndex in range(out.shape[1]):
            out[rowIndex, colIndex] = mat[colIndex, rowIndex]
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef reluGrad(double[:] vec, double[:] out):
    # Note: vec is the output i.e vec = relu(x)
    cdef int index
    for index in range(out.shape[0]):
        out[index] = 1.0 * (vec[index] > 0)
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef tanhvGrad(double[:] vec, double[:] out):
    # Note: vec is the output i.e vec = tanhv(x)
    cdef int index
    for index in range(out.shape[0]):
        out[index] = 1 - vec[index] * vec[index]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef elemMul(double[:] vecA, double[:] vecB, double[:] out):
    cdef int index
    for index in range(out.shape[0]):
        out[index] = vecA[index] * vecB[index]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef cartProd(double[:] vecA, double[:] vecB, double[:, :] out):
    cdef int colIndex, rowIndex
    for rowIndex in range(out.shape[0]):
        for colIndex in range(out.shape[1]):
            out[rowIndex, colIndex] = vecA[rowIndex] * vecB[colIndex]
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef scalarMul(double[:] vec, double scalar, double[:] out):
    cdef int index
    for index in range(out.shape[0]):
        out[index] = scalar * vec[index]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef scalarMulMat(double[:, :] mat, double scalar, double[:, :] out):
    cdef int colIndex, rowIndex
    for rowIndex in range(out.shape[0]):
        for colIndex in range(out.shape[1]):
            out[rowIndex, colIndex] = scalar * mat[rowIndex, colIndex]


cdef class Net:
    cpdef forward(self, double[:] vec):
        raise ValueError()
        
    cpdef backward(self, double[:] vec):
        raise ValueError()
        
    cpdef update(self,  double learningRate, double momentumDecay):
        raise ValueError()

cdef class ReluLinear(Net):
    cdef public double[:,:] inToHiddenMat 
    cdef public double[:] inToHiddenBias
    cdef public double[:,:] hiddenToOutMat 
    cdef public double[:] hiddenToOutBias
    cdef public double[:] hidden 
    cdef public double[:] out 
    cdef public double[:] vec
    cdef public double[:,:] hiddenToOutMat_T 
    cdef public double[:] postHiddenBackGrad
    cdef public double[:] outBackGrad
    cdef public double[:] preHiddenBackGrad 
    cdef public double[:,:] inToHiddenMat_T 
    cdef public double[:] grad 
    cdef public double[:,:] hiddenToOutMatGrad
    cdef public double[:] hiddenToOutBiasGrad 
    cdef public double[:] inToHiddenBiasGrad
    cdef public double[:,:] inToHiddenMatGrad
    cdef public double[:,:] hiddenToOutMatGradSum
    cdef public double[:] hiddenToOutBiasGradSum
    cdef public double[:] inToHiddenBiasGradSum
    cdef public double[:,:] inToHiddenMatGradSum 
    cdef public double[:,:] hiddenToOutMatGradMom
    cdef public double[:] hiddenToOutBiasGradMom 
    cdef public double[:] inToHiddenBiasGradMom
    cdef public double[:,:] inToHiddenMatGradMom 
    cdef public double[:,:] hiddenToOutMatUpdate
    cdef public double[:] hiddenToOutBiasUpdate
    cdef public double[:] inToHiddenBiasUpdate
    cdef public double[:,:] inToHiddenMatUpdate
    cdef public object npInToHiddenMat
    cdef public object npInToHiddenBias
    cdef public object npHiddenToOutMat
    cdef public object npHiddenToOutBias
    cdef public object npHidden
    cdef public object npOut
    cdef public object npVec
    cdef public object npHiddenToOutMat_T
    cdef public object npPostHiddenBackGrad
    cdef public object npOutBackGrad
    cdef public object npPreHiddenBackGrad
    cdef public object npInToHiddenMat_T
    cdef public object npGrad
    cdef public object npHiddenToOutMatGrad
    cdef public object npHiddenToOutBiasGrad
    cdef public object npInToHiddenBiasGrad
    cdef public object npInToHiddenMatGrad
    cdef public object npHiddenToOutMatGradSum
    cdef public object npHiddenToOutBiasGradSum
    cdef public object npInToHiddenBiasGradSum
    cdef public object npInToHiddenMatGradSum
    cdef public object npHiddenToOutMatGradMom
    cdef public object npHiddenToOutBiasGradMom
    cdef public object npInToHiddenBiasGradMom
    cdef public object npInToHiddenMatGradMom
    cdef public object npHiddenToOutMatUpdate
    cdef public object npHiddenToOutBiasUpdate
    cdef public object npInToHiddenBiasUpdate
    cdef public object npInToHiddenMatUpdate
        
    def __init__(self, inCount, hiddenCount, outCount):
        # XAVIER INITIALIZATION
        stdev =  (3/ inCount) ** 0.5
        self.npInToHiddenMat = np.random.uniform(-stdev, stdev, (hiddenCount, inCount))
        self.npInToHiddenBias = np.random.uniform(-stdev, stdev, hiddenCount)
        stdev = (3/ hiddenCount) ** 0.5
        self.npHiddenToOutMat = np.random.uniform(-stdev, stdev, (outCount, hiddenCount))
        self.npHiddenToOutBias = np.random.uniform(-stdev, stdev, outCount)
        
        # Python Array Initialization
        self.npHidden = np.zeros(hiddenCount)
        self.npOut = np.zeros(outCount)
        self.npVec = np.zeros(inCount)
        self.npHiddenToOutMat_T = np.zeros((hiddenCount, outCount))
        self.npPostHiddenBackGrad = np.zeros(hiddenCount)
        self.npOutBackGrad = np.ones(outCount)
        self.npPreHiddenBackGrad = np.zeros(hiddenCount)
        self.npInToHiddenMat_T = np.zeros((inCount, hiddenCount))
        self.npGrad = np.zeros(inCount)
        self.npHiddenToOutMatGrad = np.zeros((outCount, hiddenCount))
        self.npHiddenToOutBiasGrad = np.zeros((outCount))
        self.npInToHiddenBiasGrad = np.zeros((hiddenCount))
        self.npInToHiddenMatGrad = np.zeros((hiddenCount, inCount))
        self.npHiddenToOutMatGradSum = np.zeros((outCount, hiddenCount))
        self.npHiddenToOutBiasGradSum = np.zeros((outCount))
        self.npInToHiddenBiasGradSum = np.zeros((hiddenCount))
        self.npInToHiddenMatGradSum = np.zeros((hiddenCount, inCount))
        self.npHiddenToOutMatGradMom = np.zeros((outCount, hiddenCount))
        self.npHiddenToOutBiasGradMom = np.zeros((outCount))
        self.npInToHiddenBiasGradMom = np.zeros((hiddenCount))
        self.npInToHiddenMatGradMom = np.zeros((hiddenCount, inCount))
        self.npHiddenToOutMatUpdate = np.zeros((outCount, hiddenCount))
        self.npHiddenToOutBiasUpdate = np.zeros((outCount))
        self.npInToHiddenBiasUpdate = np.zeros((hiddenCount))
        self.npInToHiddenMatUpdate = np.zeros((hiddenCount, inCount))
        
        # Memorview Initialization
        self.inToHiddenMat = self.npInToHiddenMat
        self.inToHiddenBias = self.npInToHiddenBias
        self.hiddenToOutMat = self.npHiddenToOutMat
        self.hiddenToOutBias = self.npHiddenToOutBias
        self.hidden = self.npHidden
        self.out = self.npOut
        self.vec = self.npVec
        self.hiddenToOutMat_T = self.npHiddenToOutMat_T
        self.postHiddenBackGrad = self.npPostHiddenBackGrad
        self.outBackGrad = self.npOutBackGrad
        self.preHiddenBackGrad = self.npPreHiddenBackGrad
        self.inToHiddenMat_T = self.npInToHiddenMat_T
        self.grad = self.npGrad
        self.hiddenToOutMatGrad = self.npHiddenToOutMatGrad
        self.hiddenToOutBiasGrad = self.npHiddenToOutBiasGrad
        self.inToHiddenBiasGrad = self.npInToHiddenBiasGrad
        self.inToHiddenMatGrad = self.npInToHiddenMatGrad
        self.hiddenToOutMatGradSum = self.npHiddenToOutMatGradSum
        self.hiddenToOutBiasGradSum = self.npHiddenToOutBiasGradSum
        self.inToHiddenBiasGradSum = self.npInToHiddenBiasGradSum
        self.inToHiddenMatGradSum = self.npInToHiddenMatGradSum
        self.hiddenToOutMatGradMom = self.npHiddenToOutMatGradMom
        self.hiddenToOutBiasGradMom = self.npHiddenToOutBiasGradMom
        self.inToHiddenBiasGradMom = self.npInToHiddenBiasGradMom
        self.inToHiddenMatGradMom = self.npInToHiddenMatGradMom
        self.hiddenToOutMatUpdate = self.npHiddenToOutMatUpdate
        self.hiddenToOutBiasUpdate = self.npHiddenToOutBiasUpdate
        self.inToHiddenBiasUpdate = self.npInToHiddenBiasUpdate
        self.inToHiddenMatUpdate = self.npInToHiddenMatUpdate
        

        # Set Transposes of Matrices
        transpose(self.inToHiddenMat, self.inToHiddenMat_T)
        transpose(self.hiddenToOutMat, self.hiddenToOutMat_T)

    cpdef forward(self, double[:] vec):
        self.vec[:] = vec
        matMul(self.inToHiddenMat, vec, self.hidden)
        add(self.hidden, self.inToHiddenBias, self.hidden)
        relu(self.hidden, self.hidden)
        matMul(self.hiddenToOutMat, self.hidden, self.out)
        add(self.out, self.hiddenToOutBias, self.out)
        
    cpdef calcGrad(self):
        matMul(self.hiddenToOutMat_T, self.outBackGrad, self.postHiddenBackGrad)
        reluGrad(self.hidden, self.preHiddenBackGrad)
        elemMul(self.preHiddenBackGrad, self.postHiddenBackGrad, self.preHiddenBackGrad)
        matMul(self.inToHiddenMat_T, self.preHiddenBackGrad, self.grad)
        
    cpdef backward(self, double[:] backGrad):
        # Get Hidden Layer Gradient
        cartProd(backGrad, self.hidden, self.hiddenToOutMatGrad)
        self.hiddenToOutBiasGrad[:] = backGrad
        
        # Get Input Layer Gradient
        matMul(self.hiddenToOutMat_T, backGrad, self.postHiddenBackGrad)
        reluGrad(self.hidden, self.preHiddenBackGrad)
        elemMul(self.preHiddenBackGrad, self.postHiddenBackGrad, self.preHiddenBackGrad)
        cartProd(self.preHiddenBackGrad, self.vec, self.inToHiddenMatGrad)
        self.inToHiddenBiasGrad[:] = self.preHiddenBackGrad
        
        # Add Gradients to a Sum
        addMat(self.hiddenToOutMatGradSum, self.hiddenToOutMatGrad, self.hiddenToOutMatGradSum)
        add(self.hiddenToOutBiasGradSum, self.hiddenToOutBiasGrad, self.hiddenToOutBiasGradSum)
        add(self.inToHiddenBiasGradSum, self.inToHiddenBiasGrad, self.inToHiddenBiasGradSum)
        addMat(self.inToHiddenMatGradSum, self.inToHiddenMatGrad, self.inToHiddenMatGradSum)
        
    cpdef update(self, double learningRate, double momentumDecay):
        # Update InToHidden Matrix
        scalarMulMat(self.inToHiddenMatGradMom, momentumDecay, self.inToHiddenMatGradMom)
        scalarMulMat(self.inToHiddenMatGradSum, 1- momentumDecay, self.inToHiddenMatGradSum)
        addMat(self.inToHiddenMatGradSum, self.inToHiddenMatGradMom, self.inToHiddenMatGradMom)
        scalarMulMat(self.inToHiddenMatGradMom, learningRate, self.inToHiddenMatUpdate)
        addMat(self.inToHiddenMatUpdate, self.inToHiddenMat, self.inToHiddenMat)

        # Update InToHidden Bias
        scalarMul(self.inToHiddenBiasGradMom, momentumDecay, self.inToHiddenBiasGradMom)
        scalarMul(self.inToHiddenBiasGradSum, 1- momentumDecay, self.inToHiddenBiasGradSum)
        add(self.inToHiddenBiasGradSum, self.inToHiddenBiasGradMom, self.inToHiddenBiasGradMom)
        scalarMul(self.inToHiddenBiasGradMom, learningRate, self.inToHiddenBiasUpdate)
        add(self.inToHiddenBiasUpdate, self.inToHiddenBias, self.inToHiddenBias)
        
        # Update HiddenToOut Matrix
        scalarMulMat(self.hiddenToOutMatGradMom, momentumDecay, self.hiddenToOutMatGradMom)
        scalarMulMat(self.hiddenToOutMatGradSum, 1- momentumDecay, self.hiddenToOutMatGradSum)
        addMat(self.hiddenToOutMatGradSum, self.hiddenToOutMatGradMom, self.hiddenToOutMatGradMom)
        scalarMulMat(self.hiddenToOutMatGradMom, learningRate, self.hiddenToOutMatUpdate)
        addMat(self.hiddenToOutMatUpdate, self.hiddenToOutMat, self.hiddenToOutMat)

        # Update HiddenToOut Bias
        scalarMul(self.hiddenToOutBiasGradMom, momentumDecay, self.hiddenToOutBiasGradMom)
        scalarMul(self.hiddenToOutBiasGradSum, 1- momentumDecay, self.hiddenToOutBiasGradSum)
        add(self.hiddenToOutBiasGradSum, self.hiddenToOutBiasGradMom, self.hiddenToOutBiasGradMom)
        scalarMul(self.hiddenToOutBiasGradMom, learningRate, self.hiddenToOutBiasUpdate)
        add(self.hiddenToOutBiasUpdate, self.hiddenToOutBias, self.hiddenToOutBias)
        
        # Zero All Grad Sums
        self.inToHiddenMatGradSum[:] = 0
        self.inToHiddenBiasGradSum[:] = 0
        self.hiddenToOutMatGradSum[:] = 0
        self.hiddenToOutBiasGradSum[:] = 0
        
        # Set Transposes of Matrices
        transpose(self.inToHiddenMat, self.inToHiddenMat_T)
        transpose(self.hiddenToOutMat, self.hiddenToOutMat_T)
        
    cpdef copyWeights(self, ReluLinear other):
        
        
        cdef double[:,:] newInToHiddenMat = other.npInToHiddenMat
        self.inToHiddenMat[:] = newInToHiddenMat
        cdef double[:] newInToHiddenBias = other.npInToHiddenBias
        self.inToHiddenBias[:] = newInToHiddenBias
        cdef double[:,:] newHiddenToOutMat = other.npHiddenToOutMat
        self.hiddenToOutMat[:] = newHiddenToOutMat
        cdef double[:] newHiddenToOutBias = other.npHiddenToOutBias
        self.hiddenToOutBias[:] = newHiddenToOutBias
        
        transpose(self.inToHiddenMat, self.inToHiddenMat_T)
        transpose(self.hiddenToOutMat, self.hiddenToOutMat_T)
        
cdef class SuperTanhActor(Net):
    cdef public ReluLinear reluLinearA
    cdef public ReluLinear reluLinearB
    cdef public ReluLinear reluLinearC

    cdef public double[:] preOut
    cdef public double[:] out
    cdef public double[:] noise
    cdef public double fitness

    cdef public object npPreOut
    cdef public object npOut
    cdef public object npNoise
        
    def __init__(self, inCount, hiddenCount, outCount):
        self.reluLinearA = ReluLinear(inCount, hiddenCount, outCount)
        self.reluLinearB = ReluLinear(inCount, hiddenCount, outCount)
        self.reluLinearC = ReluLinear(inCount, hiddenCount, outCount)

        self.npPreOut = np.zeros(outCount)
        self.npNoise = np.random.normal(1, 1, outCount)
        self.npOut = np.zeros(outCount)
        self.out = self.npOut

        self.preOut = self.npPreOut
        self.noise = self.npNoise
        self.fitness = 0
        
    cpdef forward(self, double[:] state):
        self.reluLinearA.forward(state)
        self.reluLinearB.forward(state)
        self.reluLinearC.forward(state)
        sub(self.reluLinearB.out, self.reluLinearC.out, self.preOut)
        elemMul(self.noise, self.preOut, self.preOut)
        add(self.reluLinearA.out, self.preOut, self.preOut)
        tanhv(self.preOut, self.out)    
        
    cpdef backward(self, double[:] backGrad):
        raise ValueError()
        
    cpdef update(self,  double learningRate, double momentumDecay):
        raise ValueError()
        
    cpdef get_action(self, double[:] state):
        self.reluLinearA.forward(state)
        self.reluLinearB.forward(state)
        self.reluLinearC.forward(state)
        sub(self.reluLinearB.out, self.reluLinearC.out, self.out)
        elemMul(self.noise, self.out, self.out)
        add(self.reluLinearA.out, self.out, self.out)
        tanhv(self.out, self.out)
        return self.npOut
        
    cpdef reset(self, TanhActor a, TanhActor b, TanhActor c):
        npNewNoise = np.random.normal(1, 1, self.npNoise.shape[0])
        cdef double[:] newNoise = npNewNoise
        self.noise[:] = newNoise
        
        self.reluLinearA.copyWeights(a.reluLinear)
        self.reluLinearB.copyWeights(b.reluLinear)
        self.reluLinearC.copyWeights(c.reluLinear)
        
        
        
cdef class TanhActor(Net):
    cdef public ReluLinear reluLinear
    cdef public double[:] out
    cdef public double[:] preOut
    cdef public double[:] preOutBackGrad
    cdef public object npPreOut
    cdef public object npOut
    cdef public object npPreOutBackGrad
    cdef public double fitness
    
    
    def __init__(self, inCount, hiddenCount, outCount):
        self.reluLinear = ReluLinear(inCount, hiddenCount, outCount)
        self.npOut = np.zeros(outCount)
        self.npPreOut = np.zeros(outCount)
        self.npPreOutBackGrad = np.zeros(outCount)
        self.out = self.npOut
        self.preOut = self.npPreOut
        self.preOutBackGrad = self.npPreOutBackGrad
        self.fitness = 0

    cpdef forward(self, double[:] vec):
        self.reluLinear.forward(vec)
        self.preOut[:] = self.reluLinear.out
        tanhv(self.preOut, self.out)

    cpdef get_action(self, double[:] state):
        self.reluLinear.forward(state)
        tanhv(self.reluLinear.out, self.out)
        return self.npOut
        
    cpdef backward(self, double[:] backGrad):
        tanhvGrad(self.reluLinear.out, self.preOutBackGrad)
        elemMul(self.preOutBackGrad, backGrad, self.preOutBackGrad)
        self.reluLinear.backward(self.preOutBackGrad)
        
    cpdef update(self, double learningRate, double momentumDecay):
        self.reluLinear.update(learningRate, momentumDecay)

        
def initDpg(data):
    stateCount = 8
    actionCount = 2
    criticHiddenCount = data['Critic Hidden Count']
    agentHiddenCount = data['Actor Hidden Count']
    policyCount = data['Trains per Episode']
    number_agents = data['Number of Agents']
    
    criticCol = [ReluLinear(stateCount + actionCount, criticHiddenCount, 1) \
        for i in range(number_agents)]
    subPopulationCol = [[TanhActor(stateCount, agentHiddenCount, actionCount) \
        for i in range(policyCount//2)] for j in range(number_agents)] 
    superPopulationCol = [[SuperTanhActor(stateCount, agentHiddenCount, actionCount) \
        for i in range(policyCount - policyCount//2)] for j in range(number_agents)] 
    populationCol = [None for j in range(number_agents)] 
    for agentIndex in range(number_agents):
        populationCol[agentIndex] = subPopulationCol[agentIndex] + superPopulationCol[agentIndex]
    data['Agent Populations'] = populationCol
    data['Agent Super Populations'] = superPopulationCol
    data['Agent Sub Populations'] = subPopulationCol
    data['Agent Critics'] = criticCol
    
        
def initDpgCriticOnly(data):
    stateCount = 8
    actionCount = 2
    criticHiddenCount = data['Critic Hidden Count']
    number_agents = data['Number of Agents']
    
    criticCol = [ReluLinear(stateCount + actionCount, criticHiddenCount, 1) \
        for i in range(number_agents)]
    data['Agent Critics'] = criticCol
    
def rewardAgentsWithCritic(data):
    cdef int number_agents = data['Number of Agents']
    cdef int policyIndex = data["World Index"]
    cdef int stepCount = data["Steps"]
    criticCol = data['Agent Critics']
    populationCol = data['Agent Populations']
    cdef ReluLinear critic
    cdef double[:, :, :, :] replay = data['Experience Replay']
    cdef int stateCount = 8
    cdef int actionCount = 2
    
    cdef int agentIndex, stepIndex
    npAgentReward = np.ones(number_agents) * float("-inf")
    cdef double[:] agentReward = npAgentReward
    npStateAction = np.zeros(stateCount + actionCount)
    cdef double[:] stateAction = npStateAction
    
    # Begin update for each agent
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        for stepIndex in range(stepCount):
            stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                :stateCount + actionCount]
            critic.forward(stateAction)
            agentReward[agentIndex] = max(critic.out[0], agentReward[agentIndex])
                
    data["Agent Rewards"] = npAgentReward
    
def rewardAgentsWithCritic2(data):
    cdef int number_agents = data['Number of Agents']
    cdef int policyIndex = data["World Index"]
    cdef int stepCount = data["Steps"]
    criticCol = data['Agent Critics']
    populationCol = data['Agent Populations']
    cdef ReluLinear critic
    cdef double[:, :, :, :] replay = data['Experience Replay']
    cdef int stateCount = 8
    cdef int actionCount = 2
    
    cdef int agentIndex, stepIndex
    npAgentReward = np.zeros(number_agents) 
    cdef double[:] agentReward = npAgentReward
    npStateAction = np.zeros(stateCount + actionCount)
    cdef double[:] stateAction = npStateAction
    
    # Begin update for each agent
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        for stepIndex in range(stepCount):
            stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                :stateCount + actionCount]
            critic.forward(stateAction)
            agentReward[agentIndex] += critic.out[0]
                
    data["Agent Rewards"] = npAgentReward
    
def assignActors(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[agentIndex][worldIndex]
    data["Agent Policies"] = policyCol
    
def assignSubActor0(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Sub Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for agentIndex in range(number_agents):
        policyCol[agentIndex] = populationCol[agentIndex][0]
    data["Agent Policies"] = policyCol
    
    
def shuffleActors(data):
    number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    for agentIndex in range(number_agents):
        random.shuffle(populationCol[agentIndex])
    data['Agent Populations'] = populationCol
            
cpdef updateCritics(data):
    cdef int number_agents = data['Number of Agents']
    cdef int policyCount = data["Trains per Episode"]
    cdef int stepCount = data["Steps"]
    cdef int batchCount = data["Critic Batch Count"]
    cdef int samplePerBatchCount = data["Critic Sample Count Per Batch"]
    cdef double learningRate = data["Critic Learning Rate"]
    cdef double momentumDecay = data["Critic Momentum Decay"]
    cdef int sampleCount = batchCount * samplePerBatchCount
    criticCol = data['Agent Critics']
    cdef ReluLinear critic
    cdef double[:, :, :, :] replay = data['Experience Replay']
    cdef int stateCount = 8
    cdef int actionCount = 2
    
    cdef int agentIndex, batchIndex, samplePerBatchIndex, sampleIndex
    cdef int policyIndex, stepIndex
    npStateAction = np.zeros(stateCount + actionCount)
    npError = np.zeros(1)
    npErrorCol = np.zeros((number_agents, policyCount))
    npReward = np.zeros(1)
    cdef double[:] stateAction = npStateAction
    cdef double[:] reward = npReward
    cdef double[:] error = npError
    cdef double[:, :] errorCol = npErrorCol
    
    cdef double maxOut
    
    # Get all random numbers with only two python (numpy) call for efficiency
    npRandomPolicyIndexBuf = np.random.randint(0, policyCount, sampleCount, dtype = np.intc)
    npRandomStepIndexBuf= np.random.randint(0, stepCount, sampleCount, dtype = np.intc)
    cdef int[:] randomPolicyIndexBuf = npRandomPolicyIndexBuf
    cdef int[:] randomStepIndexBuf = npRandomStepIndexBuf
    
    # Begin update for each agent
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        
        # Keep a total sample index count for random number buffers
        sampleIndex = 0
        
        # Update in batches
        for batchIndex in range(batchCount):
            for samplePerBatchIndex in range(samplePerBatchCount):
                policyIndex = randomPolicyIndexBuf[sampleIndex]
                stepIndex = randomStepIndexBuf[sampleIndex]
                stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                    :stateCount + actionCount]
                reward[:] = replay[agentIndex, policyIndex, stepIndex, \
                    stateCount + actionCount:]
                critic.forward(stateAction)
                sub(reward, critic.out, error)
                critic.backward(error)
                sampleIndex += 1
            critic.update(learningRate, momentumDecay)
            
    # Establish all error
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        for policyIndex in range(policyCount):
            rewardEstimate = 0
            reward[0] = replay[agentIndex, policyIndex, 0, \
                    stateCount + actionCount]
            maxOut = float("-inf")
            for stepIndex in range(stepCount):
                stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                    :stateCount + actionCount]
                critic.forward(stateAction)
                rewardEstimate += critic.out[0]
                maxOut = max(maxOut, critic.out[0])
            # errorCol[agentIndex, policyIndex] = reward[0] - rewardEstimate/stepCount 
            errorCol[agentIndex, policyIndex] = reward[0] - maxOut    
            
    data["Critic Loss"] = np.sum(-npErrorCol**2, axis = 1)

cpdef updateCritics2(data):
    cdef int number_agents = data['Number of Agents']
    cdef int policyCount = data["Trains per Episode"]
    cdef int stepCount = data["Steps"]
    cdef int batchCount = data["Critic Batch Count"]
    cdef int samplePerBatchCount = data["Critic Sample Count Per Batch"]
    cdef double learningRate = data["Critic Learning Rate"]
    cdef double momentumDecay = data["Critic Momentum Decay"]
    cdef int sampleCount = batchCount * samplePerBatchCount
    criticCol = data['Agent Critics']
    cdef ReluLinear critic
    cdef double[:, :, :, :] replay = data['Experience Replay']
    cdef int stateCount = 8
    cdef int actionCount = 2
    
    cdef int agentIndex, batchIndex, samplePerBatchIndex, sampleIndex
    cdef int policyIndex, stepIndex
    npStateAction = np.zeros(stateCount + actionCount)
    npError = np.zeros(1)
    npErrorCol = np.zeros((number_agents, policyCount))
    cdef double[:] stateAction = npStateAction
    cdef double[:] error = npError
    cdef double[:, :] errorCol = npErrorCol
    cdef double reward, rewardEstimate
    
    # Get all random numbers with only two python (numpy) call for efficiency
    npRandomPolicyIndexBuf = np.random.randint(0, policyCount, sampleCount, dtype = np.intc)
    npRandomStepIndexBuf= np.random.randint(0, stepCount, sampleCount, dtype = np.intc)
    cdef int[:] randomPolicyIndexBuf = npRandomPolicyIndexBuf
    cdef int[:] randomStepIndexBuf = npRandomStepIndexBuf
    
    # Establish all error
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        for policyIndex in range(policyCount):
            rewardEstimate = 0
            reward = replay[agentIndex, policyIndex, 0, \
                    stateCount + actionCount]
            for stepIndex in range(stepCount):
                stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                    :stateCount + actionCount]
                critic.forward(stateAction)
                rewardEstimate += critic.out[0]
            errorCol[agentIndex, policyIndex] = reward - rewardEstimate

                    
    
    # Begin update for each agent
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        
        # Keep a total sample index count for random number buffers
        sampleIndex = 0
        
        # Update in batches
        for batchIndex in range(batchCount):
            for samplePerBatchIndex in range(samplePerBatchCount):
                policyIndex = randomPolicyIndexBuf[sampleIndex]
                stepIndex = randomStepIndexBuf[sampleIndex]
                stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                    :stateCount + actionCount]
                error[0] = errorCol[agentIndex, policyIndex]
                critic.forward(stateAction)
                critic.backward(error)
                sampleIndex += 1
            critic.update(learningRate, momentumDecay)
            
    data["Critic Loss"] = np.sum(-npErrorCol**2, axis = 1)

# TODO            
# cpdef calcCriticLoss(data):
#     cdef int number_agents = data['Number of Agents']
#     cdef int policyCount = data["Trains per Episode"]
#     cdef int stepCount = data["Steps"]
#     criticCol = data['Agent Critics']
#     cdef ReluLinear critic
#     cdef double[:, :, :, :] replay = data['Experience Replay']
#     cdef int stateCount = 8
#     cdef int actionCount = 2
#     
#     cdef int agentIndex, policyIndex, stepIndex
#     npStateAction = np.zeros(stateCount + actionCount)
#     npError = np.zeros(1)
#     npReward = np.zeros(1)
#     cdef double[:] stateAction = npStateAction
#     cdef double[:] reward = npReward
#     cdef double[:] error = npError
#     
#     
#     npLoss = np.zeros(number_agents)
#     cdef double[:] loss = npLoss
#     
#     # Begin loss calculation for each agent
#     for agentIndex in range(number_agents):
#         critic = criticCol[agentIndex]
#         for policyIndex in range(policyCount):
#             for stepIndex in range(stepCount):
#                 
#                 stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
#                     :stateCount + actionCount]
#                 reward[:] = replay[agentIndex, policyIndex, stepIndex, \
#                     stateCount + actionCount:]
#                 critic.forward(stateAction)
#                 sub(reward, critic.out, error)
#                 loss[agentIndex] += error[0] * error[0]
#                 
#     data["Critic Loss"] = npLoss
                
                
cpdef updateActors(data):
    cdef int number_agents = data['Number of Agents']
    cdef int policyCount = data["Trains per Episode"]
    cdef int stepCount = data["Steps"]
    cdef int batchCount = data["Actor Batch Count"]
    cdef int samplePerBatchCount = data["Actor Sample Count Per Batch"]
    cdef double learningRate = data["Actor Learning Rate"]
    cdef double momentumDecay = data["Actor Momentum Decay"]
    cdef int sampleCount = batchCount * samplePerBatchCount
    criticCol = data['Agent Critics']
    populationCol = data['Agent Populations']
    cdef ReluLinear critic
    cdef TanhActor actor
    cdef double[:, :, :, :] replay = data['Experience Replay']
    cdef int stateCount = 8
    cdef int actionCount = 2
    
    cdef int agentIndex, policyIndex, stepIndex, sampleIndex
    npStateAction = np.zeros(stateCount + actionCount)
    npState = np.zeros(stateCount)
    npActionGrad = np.zeros(actionCount)
    cdef double[:] stateAction = npStateAction
    cdef double[:] state = npState
    cdef double[:] actionGrad = npActionGrad
    npRandomStepIndexBuf= np.random.randint(0, stepCount, sampleCount, dtype = np.intc)
    cdef int[:] randomStepIndexBuf = npRandomStepIndexBuf
    
    
    # Begin update for each agent
    for agentIndex in range(number_agents):
        critic = criticCol[agentIndex]
        
        for policyIndex in range(policyCount):
            actor = populationCol[agentIndex][policyIndex]
            
            # Keep a total sample index count for random number buffers
            sampleIndex = 0
        
            # Update in batches
            for batchIndex in range(batchCount):
                for samplePerBatchIndex in range(samplePerBatchCount):
                    stepIndex = randomStepIndexBuf[sampleIndex]
                    stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
                        :stateCount + actionCount]
                    state[:] = replay[agentIndex, policyIndex, stepIndex, \
                        :stateCount]
                    critic.forward(stateAction)
                    actor.forward(state)
                    critic.calcGrad()
                    actionGrad = critic.grad[stateCount: stateCount + actionCount]
                    actor.backward(actionGrad)
                    sampleIndex += 1
                actor.update(learningRate, momentumDecay)
    
# TODO    
# cpdef scoreAgentsWithCritics(data):
#     cdef int number_agents = data['Number of Agents']
#     cdef int policyCount = data["Trains per Episode"]
#     cdef int stepCount = data["Steps"]
#     criticCol = data['Agent Critics']
#     populationCol = data['Agent Populations']
#     cdef ReluLinear critic
#     cdef double[:, :, :, :] replay = data['Experience Replay']
#     cdef int stateCount = 8
#     cdef int actionCount = 2
#     
#     cdef int agentIndex, policyIndex, stepIndex
#     npScore = np.zeros(number_agents)
#     cdef double[:] score = npScore
#     npStateAction = np.zeros(stateCount + actionCount)
#     npAction = np.zeros(actionCount)
#     npState = np.zeros(stateCount)
#     cdef double[:] action = npAction
#     cdef double[:] state = npState
#     cdef double[:] stateAction = npStateAction
#     
#     # Begin update for each agent
#     for agentIndex in range(number_agents):
#         critic = criticCol[agentIndex]
#         for policyIndex in range(policyCount):
#             policy = populationCol[agentIndex][policyIndex]
#             for stepIndex in range(stepCount):
#                 state[:] = replay[agentIndex, policyIndex, stepIndex, \
#                     :stateCount]
#                 stateAction[:] = replay[agentIndex, policyIndex, stepIndex, \
#                     :stateCount + actionCount]
#                 action = policy.get_action(state)
#                 stateAction[stateCount:stateCount + actionCount] =  \
#                     action[:]
#                 critic.forward(stateAction)
#                 score[agentIndex] += critic.out[0]
#                 
#     data["Agent Critic Score"] = npScore
  

def rewardDpgPolicies(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Agents']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(number_agents):
        policyCol[agentIndex].fitness = rewardCol[agentIndex] 
    
cpdef evolveDpgPolicies(data): 
    cdef int number_agents = data['Number of Agents']
    populationCol = data['Agent Populations']
    cdef int stepCount = data["Steps"]
    
    cdef double[:, :, :, :] replay = data['Experience Replay']
    cdef int batchCount = data["Actor Batch Count"]
    cdef int samplePerBatchCount = data["Actor Sample Count Per Batch"]
    cdef double learningRate = data["Actor Learning Rate"]
    cdef double momentumDecay = data["Actor Momentum Decay"]

    superPopulationCol = data['Agent Super Populations']
    subPopulationCol = data['Agent Sub Populations']
    cdef int sampleCount = batchCount * samplePerBatchCount
    cdef int stateCount = 8
    cdef int actionCount = 2

    cdef int agentIndex, matchIndex
    cdef int halfPopLen = int(len(populationCol[0])//2)
    cdef int superPopLen = int(len(superPopulationCol[0]))
    cdef int policyIndex, stepIndex, batchIndex, samplePerBatchIndex, sampleIndex
    npError = np.zeros(actionCount)
    npState = np.zeros(stateCount)
    cdef double[:] error = npError
    cdef double[:] state = npState
    
    cdef int winnerIndex, loserIndex
    npRandomStepIndexBuf= np.random.randint(0, stepCount, sampleCount, dtype = np.intc)
    cdef int[:] randomStepIndexBuf = npRandomStepIndexBuf
    
    for agentIndex in range(number_agents):
        population = populationCol[agentIndex]
        superPopulation = superPopulationCol[agentIndex]
        subPopulation = subPopulationCol[agentIndex]
        
        
        # Binary Tournament, mutate loser toward copy of winner, then mutate copy
        for matchIndex in range(halfPopLen):
            if population[2 * matchIndex].fitness > population[2 * matchIndex + 1].fitness:
                winnerIndex = 2 * matchIndex
                loserIndex = 2 * matchIndex + 1
            else:
                winnerIndex = 2 * matchIndex + 1
                loserIndex = 2 * matchIndex 
                
            winner = population[winnerIndex]
            loser = population[loserIndex]
            sampleIndex = 0
            if type(loser) == TanhActor:
                if type(winner) == SuperTanhActor:
                    # Update in batches
                    for batchIndex in range(batchCount):
                        for samplePerBatchIndex in range(samplePerBatchCount):
                            stepIndex = randomStepIndexBuf[sampleIndex]
                            state[:] = replay[agentIndex, loserIndex, stepIndex, \
                                :stateCount]
                            winner.forward(state)
                            loser.forward(state)
                            sub(winner.out, loser.out, error)
                            loser.backward(error)
                            sampleIndex += 1
                        loser.update(learningRate, momentumDecay)
                else:
                    loser.reluLinear.copyWeights(winner.reluLinear)
            else:
                parents = np.random.choice(subPopulation, 3, False)
                loser.reset(parents[0], parents[1], parents[2])

        data['Agent Populations'][agentIndex] = population
    