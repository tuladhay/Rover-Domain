"""
Provides Rover Domain Agents' Sensing, Processing and Movement Dynamics

Writes:
"Agent Observations": (ArrayLike[agentCount:8]<double>)
    <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
"Agent Actions": (ArrayLike[agentCount:2]<double>)
"Agent Positions": (ArrayLike[agentCount:2]<double>)
"Agent Orientations": (ArrayLike[agentCount:2]<double>)

Reads:
'Number of Agents' (int)
'Number of POIs' (int)
"Distance Metric Lower Limit" (double)
"Agent Positions" (ArrayLike[agentCount:2]<double>)
'Poi Values' (ArrayLike<double>)
"Poi Positions" (ArrayLike[poiCount:2]<double>)
"Agent Orientations"  (ArrayLike[agentCount:2]<double>)
"Agent Policies" (ArrayLike<Policy>): A policy is any object with the function 
    get_action(ArrayLike[8]<double>) -> ArrayLike[2]<double>
"Agent Observations" (ArrayLike[agentCount:8]<double>)
"Agent Actions" (ArrayLike[agentCount:2]<double>)
"""


import numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef doAgentSense(data):
    """
     Sensor model is <aNE, aNW, aSW, aSE, pNE, pNE, pSW, pSE>
     Where a means (other) agent, p means poi, and NE,NW,SW,SE are the quadrants
    """
    cdef int agentCount = data['Number of Agents']
    cdef int poiCount = data['Number of POIs'] 
    cdef double minDistanceSqr = data["Distance Metric Lower Limit"] ** 2
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:] poiValueCol = data['Poi Values']
    cdef double[:, :] poiPositionCol = data["Poi Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    npObservationCol = np.zeros((agentCount, 8), dtype = np.float64)
    cdef double[:, :] observationCol = npObservationCol
    
    cdef int agentIndex, otherAgentIndex, poiIndex, obsIndex
    cdef double globalFrameSeparation0, globalFrameSeparation1
    cdef double agentFrameSeparation0, agentFrameSeparation1

    cdef double distanceSqr
    
    
    for agentIndex in range(agentCount):

        # calculate observation values due to other agents
        for otherAgentIndex in range(agentCount):
            
            # agents do not sense self (ergo skip self comparison)
            if agentIndex == otherAgentIndex:
                continue
                
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = agentPositionCol[otherAgentIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = agentPositionCol[otherAgentIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame using inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
        
            
            # other is east of agent
            if agentFrameSeparation0 > 0:
                # other is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,0] += 1.0 / distanceSqr
                else: # other is south-east of agent
                    observationCol[agentIndex,3] += 1.0  / distanceSqr
            else:  # other is west of agent
                # other is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,1] += 1.0  / distanceSqr
                else:  # other is south-west of agent
                    observationCol[agentIndex,2] += 1.0  / distanceSqr



        # calculate observation values due to pois
        for poiIndex in range(poiCount):
            
            # Get global separation vector between the two agents    
            globalFrameSeparation0 = poiPositionCol[poiIndex,0] - agentPositionCol[agentIndex,0]
            globalFrameSeparation1 = poiPositionCol[poiIndex,1] - agentPositionCol[agentIndex,1]
            
            # Translate separation to agent frame unp.sing inverse rotation matrix
            agentFrameSeparation0 = orientationCol[agentIndex, 0] * globalFrameSeparation0 + orientationCol[agentIndex, 1] * globalFrameSeparation1 
            agentFrameSeparation1 = orientationCol[agentIndex, 0] * globalFrameSeparation1 - orientationCol[agentIndex, 1] * globalFrameSeparation0 
            distanceSqr = agentFrameSeparation0 * agentFrameSeparation0 + agentFrameSeparation1 * agentFrameSeparation1
            
            # By bounding distance value we implicitly bound sensor values
            if distanceSqr < minDistanceSqr:
                distanceSqr = minDistanceSqr
            
            # poi is east of agent
            if agentFrameSeparation0> 0:
                # poi is north-east of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,4] += poiValueCol[poiIndex]  / distanceSqr
                else: # poi is south-east of agent
                    observationCol[agentIndex,7] += poiValueCol[poiIndex]  / distanceSqr
            else:  # poi is west of agent
                # poi is north-west of agent
                if agentFrameSeparation1 > 0:
                    observationCol[agentIndex,5] += poiValueCol[poiIndex]  / distanceSqr
                else:  # poi is south-west of agent
                    observationCol[agentIndex,6] += poiValueCol[poiIndex]  / distanceSqr
                    
    data["Agent Observations"] = npObservationCol

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cpdef doAgentProcess(data):
    """
    Get action for each agent using their policy and their observation
    """
    cdef int agentCount = data['Number of Agents']
    actionCol = np.zeros((agentCount, 2), dtype = np.float_)
    policyCol = data["Agent Policies"]
    observationCol = data["Agent Observations"]
    cdef int agentIndex
    for agentIndex in range(agentCount):
        actionCol[agentIndex] = policyCol[agentIndex].get_action(observationCol[agentIndex])
    data["Agent Actions"] = actionCol

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.  
cpdef doAgentMove(data):
    """
    Take each agent's action in the world, clip it between -1 and 1, then use
        the result to simultaneously move and reioreient the agent in the world.
    """
    cdef int agentCount = data['Number of Agents']
    cdef double[:, :] agentPositionCol = data["Agent Positions"]
    cdef double[:, :] orientationCol = data["Agent Orientations"]
    npActionCol = np.array(data["Agent Actions"]).astype(np.float_)
    npActionCol = np.clip(npActionCol, -1, 1)
    cdef double[:, :] actionCol = npActionCol
    
    cdef int agentIndex

    cdef double globalFrameMotion0, globalFrameMotion1, norm
    
    # move all agents
    for agentIndex in range(agentCount):

        # turn action into global frame motion
        globalFrameMotion0 = orientationCol[agentIndex, 0] * actionCol[agentIndex, 0] - orientationCol[agentIndex, 1] * actionCol[agentIndex, 1] 
        globalFrameMotion1 = orientationCol[agentIndex, 0] * actionCol[agentIndex, 1] + orientationCol[agentIndex, 1] * actionCol[agentIndex, 0] 
        
      
        # globally move and reorient agent
        agentPositionCol[agentIndex, 0] += globalFrameMotion0
        agentPositionCol[agentIndex, 1] += globalFrameMotion1
        
        if globalFrameMotion0 == 0.0 and globalFrameMotion1 == 0.0:
            orientationCol[agentIndex,0] = 1.0
            orientationCol[agentIndex,1] = 0.0
        else:
            norm = sqrt(globalFrameMotion0**2 +  globalFrameMotion1 **2)
            orientationCol[agentIndex,0] = globalFrameMotion0 /norm
            orientationCol[agentIndex,1] = globalFrameMotion1 /norm


    data["Agent Positions"]  = agentPositionCol
    data["Agent Orientations"] = orientationCol 