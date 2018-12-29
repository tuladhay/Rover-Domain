"""
Creates blueprints (i.e. world initiationalization that should remain the same
    throughout the training generation; this is for fairer comparison between
    policies evaluated in the same generation.) 
Initializes (i.e. resets) world to blueprints' values at the start of each run

Writes:
'Agent Positions BluePrint'
'Agent Orientations BluePrint'
'Poi Positions BluePrint'
'Poi Values BluePrint'
    
Reads:
'Number of Agents'  (int)
'Setup Size'
"Agent Initialization Size": Value from 0 to 1, scaling factor relative to setup
    size for the initialization area for agents where agents are given random 
    positions and orientatitions inside this area. This area is centered in the 
    world setup. 
'Number of POIs'
'Agent Positions BluePrint'
'Agent Orientations BluePrint'
'Poi Positions BluePrint'
'Poi Values BluePrint'
'Poi Relative Static Positions'
'Poi Static Values'
"""

import numpy as np

def blueprintAgent(data):
    """
    Set/Reset agent blueprints so that agents are initialized uniform randomly
        throughout the entire setup perimeter (as if "Agent Initialization Size"
        is 1). Agents have random orientation.
    """
    agentCount = data['Number of Agents']
    setupSize = data['Setup Size']
    
    # Initialize all agents in the np.randomly in world
    data['Agent Positions BluePrint'] = np.random.rand(agentCount, 2) * [setupSize, setupSize]
    angleCol = np.random.uniform(-np.pi, np.pi, agentCount)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angleCol), np.sin(angleCol))).T

def blueprintAgentInitSize(data):
    """
    Set/Reset agent blueprints so that agents are initialized uniform randomly
        throughout perimeter defined by  "Agent Initialization Size". Agents 
        have random orientation.
    """
    agentCount = data['Number of Agents']
    setupSize = data['Setup Size']
    agentInitSize = data["Agent Initialization Size"]
    
    worldSize = np.array([setupSize, setupSize])
    
    # Initialize all agents in the np.randomly in world
    positionCol = np.random.rand(agentCount, 2) * worldSize
    positionCol *= agentInitSize
    positionCol += 0.5 * (1 - agentInitSize) * worldSize
    data['Agent Positions BluePrint'] = positionCol
    angleCol = np.random.uniform(-np.pi, np.pi, agentCount)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angleCol), np.sin(angleCol))).T

    
def blueprintPoi(data):
    """
    Set/Reset poi blueprints so that poi are initialized uniform randomly
        throughout the entire setup perimeter. Poi values start at 1 and 
        incrementally increase by 1 for each subsequent poi.
    """
    poiCount = data['Number of POIs']    
    setupSize = data['Setup Size'] 
    
    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(poiCount, 2) * [setupSize, setupSize]
    data['Poi Values BluePrint'] = np.arange(poiCount) + 1.0
 
 
def initWorld(data):
    """
    Set/Reset world data to blueprints' values
    """
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Agent Orientations'] = data['Agent Orientations BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
    data['Poi Values'] = data['Poi Values BluePrint'].copy()


def blueprintStatic(data):
    """
    Set/Reset agent and poi blueprints so that poi positions and values are 
        statically defined. Agents are initialized on top of each other in the
        exact center of the setup perimeter. Agents have random orientation.
    """
    agentCount = data['Number of Agents']
    poiCount = data['Number of POIs'] 
    setupSize = data['Setup Size']
    poiRelativeStaticPositionCol = data['Poi Relative Static Positions']
    poiStaticValues = data['Poi Static Values']
    
    data['Agent Positions BluePrint'] = np.ones((agentCount,2)) * 0.5 * [setupSize, setupSize]
    angles = np.random.uniform(-np.pi, np.pi, agentCount)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angles), np.sin(angles))).T
    data['Poi Positions BluePrint'] =  poiRelativeStaticPositionCol * [setupSize, setupSize]
    data['Poi Values BluePrint'] =  poiStaticValues.copy()
    
