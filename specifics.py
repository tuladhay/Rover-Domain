import datetime
from core import SimulationCore
import pyximport; pyximport.install() # For cython(pyx) code
from code.world_setup import * # Rover Domain Construction 
from code.agent_domain_2 import * # Rover Domain Dynamic  
from code.trajectory_history import * # Agent Position Trajectory History 
from code.reward_2 import * # Agent Reward 
from code.reward_history import * # Performance Recording 
from code.ccea_2 import * # CCEA 
from code.save_to_pickle import * # Save data as pickle file

# from code.experience_replay import *
# from code.dpg import *
    
# todo "Observation Radius" to "Activation Radius"
# todo "Miniumum Distance" to "Distance Metric Lower Limit"    
    
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
    sim.data["World Width"] = 50
    sim.data["World Length"] = 50
    sim.data["Minimum Distance"] = 1.0
    sim.data["Steps"] = 100
    sim.data["Trains per Episode"] = 50
    sim.data["Tests per Episode"] = 1
    sim.data["Number of Episodes"] = 5000
    sim.data["Specifics Name"] = "30Agents_8Poi_6Coup_Long_Comparison"
    sim.data["Mod Name"] = "global"
    
    # NOTE: all simulation core ...funcCol collections are order-sensitive
    
    # print the current Episode
    sim.testEndFuncCol.append(
        lambda data: print(data["Episode Index"], data["Global Reward"])
    )
    # sim.trialEndFuncCol.append(lambda data: print())
    
    
    # Add Rover Domain Construction Functionality
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
    sim.worldTrainBeginFuncCol.append(createTrajectoryHistories)
    sim.worldTrainStepFuncCol.append(updateTrajectoryHistories)
    sim.trialEndFuncCol.append(saveTrajectoryHistories)
    sim.worldTestBeginFuncCol.append(createTrajectoryHistories)
    sim.worldTestStepFuncCol.append(updateTrajectoryHistories)
    
    # Add Agent Training Reward Functionality
    sim.data["Coupling"] = 6
    sim.data["Observation Radius"] = 4.0
    sim.data["Reward Function"] = assignGlobalReward
    sim.worldTrainEndFuncCol.append(
        lambda data: data["Reward Function"]()
    )
    
    # Add Performance Recording Functionality
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
    sim.data["Evaluation Function"] = assignGlobalReward
    sim.worldTestEndFuncCol.append(
        lambda data: data["Evaluation Function"]()
    )
    sim.trialBeginFuncCol.append(createRewardHistory)
    sim.testEndFuncCol.append(updateRewardHistory)
    sim.trialEndFuncCol.append(saveRewardHistory)
    
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
    sim.trialBeginFuncCol.append(initCcea(input_shape= 8, num_outputs=2, num_units = 32))
    sim.worldTrainBeginFuncCol.append(assignCceaPolicies)
    sim.worldTrainEndFuncCol.append(rewardCceaPolicies)
    sim.testEndFuncCol.append(evolveCceaPolicies)
    sim.worldTestBeginFuncCol.append(assignBestCceaPolicies)
    
    
    # Save data as pickle file
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
    #sim.trialEndFuncCol.append(savePickle)
    
    return sim




