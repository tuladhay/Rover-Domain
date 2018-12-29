from rover_domain_core_gym import RoverDomainCoreGym
from mods import *
import datetime
from code.world_setup import * # Rover Domain Construction 
from code.agent_domain import * # Rover Domain Dynamic  
from code.trajectory_history import * # Agent Position Trajectory History 
from code.reward import * # Agent Reward 
from code.performance_history import * # Performance Recording 
from code.ccea import * # CCEA 
from code.save_to_pickle import * # Save data as pickle file
import random


stepCount = 5
trainCountXEpisode = 3
testCountXEpisode = 1
episodeCount =  20

# NOTE: Add the mod functions (variables) to run to modCol here:
modCol = [
    globalRewardMod,
    differenceRewardMod,
    dppRewardMod
]

i = 0
while True:
    print("Run %i"%(i))
    random.shuffle(modCol)
    for mod in modCol:
        sim = RoverDomainCoreGym()
        mod(sim)
        
        #Trial Begins
        createPerformanceHistory(sim.data)
        
        sim.data["Number of Inputs"] = 8
        sim.data["Number of Hidden Units"] = 32
        sim.data["Number of Outputs"] = 2
        sim.data['Number of Policies per Population'] = trainCountXEpisode
        initCcea(sim.data)
        
        sim.data["Number of Steps"] = stepCount
        
        for episodeIndex in range(episodeCount):
            sim.data["Episode Index"] = episodeIndex
            
            # Training Phase
            
            obs = sim.reset('Train', True)
            
            for worldIndex in range(trainCountXEpisode):
                sim.data["World Index"] = worldIndex
                obs = sim.reset('Train', False)
                assignCceaPolicies(sim.data)
                
                done = False
                stepCount = 0
                while not done:
                    doAgentProcess(sim.data)
                    jointAction = sim.data["Agent Actions"]
                    obs, reward, done, info = sim.step(jointAction)
                    stepCount += 1
                    
                rewardCceaPolicies(sim.data)
                    
                    
            # Testing Phase
                    
            obs = sim.reset('Test', True)
            assignBestCceaPolicies(sim.data)        
                    
            for worldIndex in range(testCountXEpisode):
                sim.data["World Index"] = worldIndex
                obs = sim.reset('Test', False)
                
                done = False
                stepCount = 0
                while not done:
                    doAgentProcess(sim.data)
                    jointAction = sim.data["Agent Actions"]
                    obs, reward, done, info = sim.step(jointAction)
                    stepCount += 1
                    
            evolveCceaPolicies(sim.data)
            updatePerformanceHistory(sim.data)
            
            # Trial End
            savePerformanceHistory(sim.data)
            saveTrajectoryHistory(sim.data)
        
        
        
    i += 1