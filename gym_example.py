#! /usr/bin/python3.6

from rover_domain_core_gym import rover_domain_core_gym
from mods import global_reward_mod, difference_reward_mod, dpp_reward_mod
import datetime
from code.world_setup import * # Rover Domain Construction 
from code.agent_domain import * # Rover Domain Dynamic
from code.trajectory_history import * # Agent Position Trajectory History 
from code.reward import * # Agent Reward
from code.reward_history import * # Performance Recording 
from code.ccea import * # CCEA
from code.save_to_pickle import * # Save data as pickle file
import random


step_count = 5
generations_per_episode = 3
tests_per_episode = 1
num_episodes =  20

# NOTE: Add the mod functions (variables) to run to modCol here:
modCol = [
    global_reward_mod,
    difference_reward_mod,
    dpp_reward_mod
]

i = 0
while i < 10:
    print("Run %i"%(i))
    random.shuffle(modCol)
    for mod in modCol:
        sim = rover_domain_core_gym()
        mod(sim)
        
        #Trial Begins
        createRewardHistory(sim.data)
        init_ccea(num_inputs=8, num_outputs=2, num_units=32)(sim.data)
        sim.data["Steps"] = step_count
        
        for episodeIndex in range(num_episodes):
            sim.data["Episode Index"] = episodeIndex
            
            # Training Phase
            
            obs = sim.reset('Train', True)
            
            for worldIndex in range(generations_per_episode):
                sim.data["World Index"] = worldIndex
                obs = sim.reset('Train', False)
                assign_ccea_policies(sim.data)
                
                done = False
                step_count = 0
                while not done:
                    get_agent_actions(sim.data)
                    jointAction = sim.data["Agent Actions"]
                    obs, reward, done, info = sim.step(jointAction)
                    step_count += 1
                    
                reward_ccea_policies(sim.data)
                    
                    
            # Testing Phase
                    
            obs = sim.reset('Test', True)
            assign_best_ccea_policies(sim.data)
                    
            for worldIndex in range(tests_per_episode):
                sim.data["World Index"] = worldIndex
                obs = sim.reset('Test', False)
                
                done = False
                step_count = 0
                while not done:
                    get_agent_actions(sim.data)
                    jointAction = sim.data["Agent Actions"]
                    obs, reward, done, info = sim.step(jointAction)
                    step_count += 1
                    
            evolve_ccea_policies(sim.data)
            updateRewardHistory(sim.data)
            
            # Trial End
            saveRewardHistory(sim.data)
            saveTrajectoryHistories(sim.data)

    i += 1
