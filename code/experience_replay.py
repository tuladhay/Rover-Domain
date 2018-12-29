import numpy as np

def createExperienceReplay(data):
    agentCount = data['Number of Agents']
    trainCount = data["Trains per Episode"]
    stateCount = 8
    actionCount = 2
    stepCount = data["Number of Steps"]
    
    replay = np.zeros((agentCount, trainCount, stepCount, stateCount+actionCount+1))
    
    data["Experience Replay"] = replay
    
def updateStateActionOfReplay(data):
    agentCount = data['Number of Agents']
    worldIndex = data["World Index"]
    stateCount = 8
    actionCount = 2
    observationCol = data["Agent Observations"]
    actionCol = data["Agent Actions"]
    stepIndex = data["Step Index"]
    replay = data["Experience Replay"]
    
    replay[:, worldIndex, stepIndex, :stateCount] = observationCol
    replay[:, worldIndex, stepIndex, stateCount:stateCount+actionCount] = actionCol
    
    data["Experience Replay"] = replay
    
def updateRewardOfReplay(data):
    agentCount = data['Number of Agents']
    worldIndex = data["World Index"]
    rewardCol = data["Agent Rewards"]
    stateCount = 8
    actionCount = 2
    observationCol = data["Agent Observations"]
    actionCol = data["Agent Actions"]
    stepCount = data["Number of Steps"]
    stepIndex = data["Step Index"]
    replay = data["Experience Replay"]
    
    temp = replay[:, worldIndex, :, stateCount:actionCount].T
    temp[:,:] = rewardCol
    replay[:, worldIndex, :, stateCount:actionCount] = temp.T
    
    data["Experience Replay"] = replay