import datetime
from core import SimulationCore
import pyximport; pyximport.install() # For cython(pyx) code
from code.world_setup import * # Rover Domain Construction 
from code.agent_domain import * # Rover Domain Dynamic  
from code.trajectory_history import * # Agent Position Trajectory History 
from code.reward import * # Agent Reward 
from code.performance_history import * # Performance Recording 
from code.ccea import * # CCEA 
from code.save_to_pickle import * # Save data as pickle file

# from code.experience_replay import *
# from code.dpg import *

#todo Use lambda to interface code

"""
Note the following changes:
ccea_2.pyx to ccea.pyx
reward_2.pyx to reward.pyx
agent_domain_2.pyx to agent_domain.pyx
"Observation Radius" changed to "Interaction Radius"
"Minimum Distance" to "Distance Metric Lower Limit"  
"Steps" to "Number of Steps"
4 new keys for sim.data for CCEA 
number_agents to agentCount
number_pois to poiCount
reward_history to performance_history
"Reward History" to "Performance History"
reward_history.py to performance_history.py
saveRewardHistory to savePerformanceHistory 
createRewardHistory to createPerformanceHistory
updateRewardHistory to updatePerformanceHistory 
"Performance" is new key for sim.data set to "Global Reward"
createTrajectoryHistories to createTrajectoryHistory
updateTrajectoryHistories to updateTrajectoryHistory
saveTrajectoryHistories to saveTrajectoryHistory
"World Width" to "Setup Size"
"World Length" removed
"""
    
def getSim():
    """
    Get a default rover domain simulation with some default functionality.
    Users are encouraged to modify this function and save copies of it for
     each trial to use as a parameter reference.
    
    Returns:
        SimulationCore
    """
    sim = SimulationCore()
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    
    sim.data["Number of Agents"] = 30
    sim.data["Number of POIs"] = 8
    sim.data["Distance Metric Lower Limit"] = 1.0
    sim.data["Number of Steps"] = 100
    sim.data["Trains per Episode"] = 50
    sim.data["Tests per Episode"] = 1
    sim.data["Number of Episodes"] = 5000
    sim.data["Specifics Name"] = "test"
    sim.data["Mod Name"] = "global"
    
    # NOTE: all simulation core ...funcCol collections are order-sensitive
    
    # print the current Episode
    sim.testEndFuncCol.append(
        lambda data: print(data["Episode Index"], data["Global Reward"])
    )
    # sim.trialEndFuncCol.append(lambda data: print())
    
    
    # Add Rover Domain Construction Functionality
    sim.data["Setup Size"] = 50
    sim.data['Poi Static Values'] = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0])
    sim.data['Poi Relative Static Positions'] = np.array([
        [0.0, 0.0], 
        [0.0, 1.0], 
        [1.0, 0.0], 
        [1.0, 1.0], 
        [1.0, 0.5], 
        [0.5, 1.0], 
        [0.0, 5.0],
        [0.5, 0.0]
    ])
    sim.data['Agent Initialization Size'] = 0.1
    sim.trainBeginFuncCol.append(blueprintStatic)
    sim.trainBeginFuncCol.append(blueprintAgentInitSize)
    sim.worldTrainBeginFuncCol.append(initWorld)
    sim.testBeginFuncCol.append(blueprintStatic)
    sim.testBeginFuncCol.append(blueprintAgentInitSize)
    sim.worldTestBeginFuncCol.append(initWorld)
    
    
    # Add Rover Domain Dynamic Functionality
    sim.data["Observation Function"] = doAgentSense
    sim.worldTrainStepFuncCol.append(
        lambda data: data["Observation Function"](data)
    )
    sim.worldTrainStepFuncCol.append(doAgentProcess)
    sim.worldTrainStepFuncCol.append(doAgentMove)
    sim.worldTestStepFuncCol.append(
        lambda data: data["Observation Function"](data)
    )
    sim.worldTestStepFuncCol.append(doAgentProcess)
    sim.worldTestStepFuncCol.append(doAgentMove)
    
    # Add Agent Position Trajectory History Functionality
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
    sim.worldTrainBeginFuncCol.append(createTrajectoryHistory)
    sim.worldTrainStepFuncCol.append(updateTrajectoryHistory)
    sim.trialEndFuncCol.append(saveTrajectoryHistory)
    sim.worldTestBeginFuncCol.append(createTrajectoryHistory)
    sim.worldTestStepFuncCol.append(updateTrajectoryHistory)
    
    # Add Agent Training Reward and Evaluation Functionality
    sim.data["Coupling"] = 6
    sim.data["Interaction Radius"] = 4.0
    sim.data["Reward Function"] = assignGlobalReward
    sim.data["Evaluation Function"] = assignGlobalReward
    sim.worldTrainEndFuncCol.append(
        lambda data: data["Reward Function"](data)
    )
    
    sim.worldTestEndFuncCol.append(
        lambda data: data["Evaluation Function"](data)
    )
    
    # Add Performance Recording Functionality
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
    sim.trialBeginFuncCol.append(createPerformanceHistory)
    sim.testEndFuncCol.append(updatePerformanceHistory)
    sim.trialEndFuncCol.append(savePerformanceHistory)
    
    # # Add DE Functionality (all Functionality below are dependent and are displayed together for easy accessibility)
    # from code.differential_evolution import initDe, assignDePolicies, rewardDePolicies, evolveDePolicies, assignBestDePolicies
    # sim.trialBeginFuncCol.append(initDe(input_shape= 8, num_outputs=2, num_units = 16))
    # sim.worldTrainBeginFuncCol.append(assignDePolicies)
    # sim.worldTrainEndFuncCol.append(rewardDePolicies)
    # sim.trainEndFuncCol.append(evolveDePolicies)
    # sim.worldTestBeginFuncCol.append(assignBestDePolicies)
    
    # # Experience Replay
    # sim.trialBeginFuncCol.append(createExperienceReplay)
    # sim.worldTrainStepFuncCol.append(updateStateActionOfReplay)
    # sim.worldTrainEndFuncCol.append(updateRewardOfReplay)
    
    # # Critic Network
    # sim.trialBeginFuncCol.append(initDpgCriticOnly)
    # sim.worldTrainEndFuncCol.append(rewardAgentsWithCritic)
    # #sim.trainEndFuncCol.append(lambda data: print(data["Agent Rewards"]))
    # sim.testEndFuncCol.append(updateCritics)
    # # sim.testEndFuncCol.append(calcCriticLoss)
    # sim.testEndFuncCol.append(lambda data: print(data["Critic Loss"]))
    # 
    # sim.data["Critic Batch Count"] = 25
    # sim.data["Critic Sample Count Per Batch"] = 200
    # sim.data["Critic Learning Rate"] = 0.0000005
    # sim.data["Critic Momentum Decay"] = 0.0
    # sim.data['Critic Hidden Count'] = 200
    # 
    # Add CCEA Functionality 
    sim.data["Number of Inputs"] = 8
    sim.data["Number of Hidden Units"] = 32
    sim.data["Number of Outputs"] = 2
    sim.data['Number of Policies per Population'] = sim.data["Trains per Episode"]
    sim.trialBeginFuncCol.append(initCcea)
    sim.worldTrainBeginFuncCol.append(assignCceaPolicies)
    sim.worldTrainEndFuncCol.append(rewardCceaPolicies)
    sim.testEndFuncCol.append(evolveCceaPolicies)
    sim.worldTestBeginFuncCol.append(assignBestCceaPolicies)
    
    
    # Save data as pickle file
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
    #sim.trialEndFuncCol.append(savePickle)
    
    return sim




