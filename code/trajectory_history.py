"""
This module manages agent trajectories (i.e. position and orientation)
Saving does not work well with Amazon AWS and other platforms. Might not be 
    compatible with user's operating system.

Writes:
save file to "Trajectory Save File Name"
"Agent Position History" (ArrayLike[stepCount, agentCount, 2]<double>)
"Agent Orientation Hiistory" (ArrayLike[stepCount, agentCount, 2]<double>)

Reads:
"Number of Agents" (int)
"Number of POIs" (int)
"Number of Steps" (int)
"Step Index" (int)
"Agent Positions"  (ArrayLike[agentCount, 2]<double>)
"Agent Orientations" (ArrayLike[agentCount, 2]<double>)
"Trajectory Save File Name" (string)
"Poi Positions" (ArrayLike[poiCount, 2]<double>)
"Agent Position History" (ArrayLike[stepCount, agentCount, 2]<double>)
"Agent Orientation Hiistory" (ArrayLike[stepCount, agentCount, 2]<double>)
"""


import csv
import os
import errno
import numpy as np

def createTrajectoryHistory(data):
    """
    Create a history of positions for each agent history is used to calculate 
        system performance.
    """
    agentCount = data['Number of Agents']
    historyStepCount = data["Number of Steps"] + 1
    agentPositionHistory = np.zeros((historyStepCount, agentCount, 2))
    agentOrientationHistory = np.zeros((historyStepCount, agentCount, 2))
    positionCol = data["Agent Positions"]
    orientationCol = data["Agent Orientations"]

    agentPositionHistory[0] = positionCol
    agentOrientationHistory[0] = orientationCol
    
    
    data["Agent Position History"] = agentPositionHistory
    data["Agent Orientation History"] = agentOrientationHistory
    
    
def updateTrajectoryHistory(data):
    """
    Add a new value to the history of positions for each agent
    """
    agentCount = data['Number of Agents']
    stepIndex = data["Step Index"]
    historyStepCount = data["Number of Steps"] + 1
    agentPositionHistory = data["Agent Position History"]
    agentOrientationHistory = data["Agent Orientation History"]
    positionCol = data["Agent Positions"]
    orientationCol = data["Agent Orientations"]
    
    
    agentPositionHistory[stepIndex + 1] = positionCol
    agentOrientationHistory[stepIndex + 1] = orientationCol
        
    data["Agent Position History"] = agentPositionHistory
    data["Agent Orientation History"] = agentOrientationHistory
    
def saveTrajectoryHistory(data):
    """
    Save history of position to save file
    """
    saveFileName = data["Trajectory Save File Name"]
    agentCount = data['Number of Agents']
    poiCount = data["Number of POIs"]
    historyStepCount = data["Number of Steps"] + 1
    agentPositionHistory = data["Agent Position History"]
    agentOrientationHistory = data["Agent Orientation History"]
    poiPositionCol = data["Poi Positions"]
    
    if not os.path.exists(os.path.dirname(saveFileName)):
        try:
            os.makedirs(os.path.dirname(saveFileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    with open(saveFileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for agentIndex in range(agentCount):
            writer.writerow(["Agent %d Position 0"%(agentIndex)] + [pos[0] for pos in agentPositionHistory[:,agentIndex,:]])
            writer.writerow(["Agent %d Position 1"%(agentIndex)] + [pos[1] for pos in agentPositionHistory[:,agentIndex,:]])
            writer.writerow(["Agent %d Orientation 0"%(agentIndex)] + [ori[0] for ori in agentOrientationHistory[:,agentIndex,:]])
            writer.writerow(["Agent %d Orientation 1"%(agentIndex)] + [ori[1] for ori in agentOrientationHistory[:,agentIndex,:]])
            
        for poiIndex in range(poiCount):
            writer.writerow(["Poi %d Position 0"%(poiIndex)] + [poiPositionCol[poiIndex, 0]])
            writer.writerow(["Poi %d Position 1"%(poiIndex)] + [poiPositionCol[poiIndex, 1]])
