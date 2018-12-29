"""
Evaluate Agents while simultaneously giving them a reward.

Writes:
"Global Reward" (double)
"Agent Rewards" (ArrayLike<double>)
"Performance"(double)

Reads:
'Number of Agents' (int)
'Number of POIs' (int)
"Distance Metric Lower Limit" (double)
"Number of Steps" (int)
"Coupling" (int)
"Interaction Radius" (double)
"Agent Position History" (ArrayLike[stepCount, agentCount, 2]<double>)
'Poi Values' (ArrayLike<double>)
"Poi Positions"  (ArrayLike[poiCount, 2]<double>)

"""


import numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignGlobalReward(data):
    
    cdef int agentCount = data['Number of Agents']
    cdef int poiCount = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Distance Metric Lower Limit"] ** 2
    cdef int historyStepCount = data["Number of Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double interactionRadiusSqr = data["Interaction Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
  
    
    cdef int poiIndex, stepIndex, agentIndex, observerCount
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    
    
    for poiIndex in range(poiCount):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(agentCount):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < interactionRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < interactionRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
    
    data["Global Reward"] = globalReward
    data["Agent Rewards"] = np.ones(agentCount) * globalReward
    data["Performance"] = globalReward


 
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignDifferenceReward(data):
    cdef int agentCount = data['Number of Agents']
    cdef int poiCount = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Distance Metric Lower Limit"] ** 2
    cdef int historyStepCount = data["Number of Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double interactionRadiusSqr = data["Interaction Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]

    cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    cdef double globalWithoutReward = 0.0
    
    npDifferenceRewardCol = np.zeros(agentCount)
    cdef double[:] differenceRewardCol = npDifferenceRewardCol
    
    for poiIndex in range(poiCount):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(agentCount):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < interactionRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < interactionRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr

    
    for agentIndex in range(agentCount):
        globalWithoutReward = 0
        for poiIndex in range(poiCount):
            closestObsDistanceSqr = Inf
            for stepIndex in range(historyStepCount):
                # Count how many agents observe poi, update closest distance if necessary
                observerCount = 0
                stepClosestObsDistanceSqr = Inf
                for otherAgentIndex in range(agentCount):
                    if agentIndex != otherAgentIndex:
                        # Calculate separation distance between poi and agent\
                        separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
                        separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
                        distanceSqr = separation0 * separation0 + separation1 * separation1
                        
                        # Check if agent observes poi, update closest step distance
                        if distanceSqr < interactionRadiusSqr:
                            observerCount += 1
                            if distanceSqr < stepClosestObsDistanceSqr:
                                stepClosestObsDistanceSqr = distanceSqr
                            
                            
                # update closest distance only if poi is observed    
                if observerCount >= coupling:
                    if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                        closestObsDistanceSqr = stepClosestObsDistanceSqr
            
            # add to global reward if poi is observed 
            if closestObsDistanceSqr < interactionRadiusSqr:
                if closestObsDistanceSqr < minDistanceSqr:
                    closestObsDistanceSqr = minDistanceSqr
                globalWithoutReward += poiValueCol[poiIndex] / closestObsDistanceSqr
        differenceRewardCol[agentIndex] = globalReward - globalWithoutReward
        
    data["Agent Rewards"] = npDifferenceRewardCol  
    data["Global Reward"] = globalReward
    data["Performance"] = globalReward
    

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def assignDppReward(data):
    cdef int agentCount = data['Number of Agents']
    cdef int poiCount = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Distance Metric Lower Limit"] ** 2
    cdef int historyStepCount = data["Number of Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double interactionRadiusSqr = data["Interaction Radius"] ** 2
    cdef double[:, :, :] agentPositionHistory = data["Agent Position History"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]

    
    cdef int poiIndex, stepIndex, agentIndex, observerCount, otherAgentIndex, counterfactualCount
    cdef double separation0, separation1, closestObsDistanceSqr, distanceSqr, stepClosestObsDistanceSqr
    cdef double Inf = float("inf")
    
    cdef double globalReward = 0.0
    cdef double globalWithoutReward = 0.0
    cdef double globalWithExtraReward = 0.0
    
    npDifferenceRewardCol = np.zeros(agentCount)
    cdef double[:] differenceRewardCol = npDifferenceRewardCol
    
    # Calculate Global Reward
    for poiIndex in range(poiCount):
        closestObsDistanceSqr = Inf
        for stepIndex in range(historyStepCount):
            # Count how many agents observe poi, update closest distance if necessary
            observerCount = 0
            stepClosestObsDistanceSqr = Inf
            for agentIndex in range(agentCount):
                # Calculate separation distance between poi and agent
                separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, agentIndex, 0]
                separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, agentIndex, 1]
                distanceSqr = separation0 * separation0 + separation1 * separation1
                
                # Check if agent observes poi, update closest step distance
                if distanceSqr < interactionRadiusSqr:
                    observerCount += 1
                    if distanceSqr < stepClosestObsDistanceSqr:
                        stepClosestObsDistanceSqr = distanceSqr
                        
                        
            # update closest distance only if poi is observed    
            if observerCount >= coupling:
                if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                    closestObsDistanceSqr = stepClosestObsDistanceSqr
        
        # add to global reward if poi is observed 
        if closestObsDistanceSqr < interactionRadiusSqr:
            if closestObsDistanceSqr < minDistanceSqr:
                closestObsDistanceSqr = minDistanceSqr
            globalReward += poiValueCol[poiIndex] / closestObsDistanceSqr
            
    # Calculate Difference Reward      
    for agentIndex in range(agentCount):
        globalWithoutReward = 0
        for poiIndex in range(poiCount):
            closestObsDistanceSqr = Inf
            for stepIndex in range(historyStepCount):
                # Count how many agents observe poi, update closest distance if necessary
                observerCount = 0
                stepClosestObsDistanceSqr = Inf
                for otherAgentIndex in range(agentCount):
                    if agentIndex != otherAgentIndex:
                        # Calculate separation distance between poi and agent\
                        separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
                        separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
                        distanceSqr = separation0 * separation0 + separation1 * separation1
                        
                        # Check if agent observes poi, update closest step distance
                        if distanceSqr < interactionRadiusSqr:
                            observerCount += 1
                            if distanceSqr < stepClosestObsDistanceSqr:
                                stepClosestObsDistanceSqr = distanceSqr
                            
                            
                # update closest distance only if poi is observed    
                if observerCount >= coupling:
                    if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                        closestObsDistanceSqr = stepClosestObsDistanceSqr
            
            # add to global reward if poi is observed 
            if closestObsDistanceSqr < interactionRadiusSqr:
                if closestObsDistanceSqr < minDistanceSqr:
                    closestObsDistanceSqr = minDistanceSqr
                globalWithoutReward += poiValueCol[poiIndex] / closestObsDistanceSqr
        differenceRewardCol[agentIndex] = globalReward - globalWithoutReward
    
    # Calculate Dpp Reward
    for counterfactualCount in range(coupling):
        # Calculate Difference with Extra Me Reward
        for agentIndex in range(agentCount):
            globalWithExtraReward = 0
            for poiIndex in range(poiCount):
                closestObsDistanceSqr = Inf
                for stepIndex in range(historyStepCount):
                    # Count how many agents observe poi, update closest distance if necessary
                    observerCount = 0
                    stepClosestObsDistanceSqr = Inf
                    for otherAgentIndex in range(agentCount):
                        # Calculate separation distance between poi and agent\
                        separation0 = poiPositionCol[poiIndex, 0] - agentPositionHistory[stepIndex, otherAgentIndex, 0]
                        separation1 = poiPositionCol[poiIndex, 1] - agentPositionHistory[stepIndex, otherAgentIndex, 1]
                        distanceSqr = separation0 * separation0 + separation1 * separation1
                        
                    
                        if distanceSqr < interactionRadiusSqr:
                            # Check if agent observes poi, update closest step distance
                            observerCount += 1 + ((agentIndex == otherAgentIndex) * counterfactualCount)
                            if distanceSqr < stepClosestObsDistanceSqr:
                                stepClosestObsDistanceSqr = distanceSqr

                    # update closest distance only if poi is observed    
                    if observerCount >= coupling:
                        if stepClosestObsDistanceSqr < closestObsDistanceSqr:
                            closestObsDistanceSqr = stepClosestObsDistanceSqr
                
                # add to global reward if poi is observed 
                if closestObsDistanceSqr < interactionRadiusSqr:
                    if closestObsDistanceSqr < minDistanceSqr:
                        closestObsDistanceSqr = minDistanceSqr
                    globalWithExtraReward += poiValueCol[poiIndex] / closestObsDistanceSqr
            differenceRewardCol[agentIndex] = max(differenceRewardCol[agentIndex], 
            (globalWithExtraReward - globalReward)/(1.0 + counterfactualCount))
        
    data["Agent Rewards"] = npDifferenceRewardCol  
    data["Global Reward"] = globalReward   
    data["Performance"] = globalReward

    