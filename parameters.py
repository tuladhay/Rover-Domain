import pyximport; pyximport.install()  # For cython(pyx) code
import numpy as np
from code.reward import calc_global_reward, calc_difference_reward, calc_dpp_reward

"""
data:
A dictionary shared amongst all functions in the simulation.
User may add any property they may want to have shared by all provided functions
SimulationCore provides and manages the following keys during run() execution:
    "Steps": duration of world measured in time steps,
    "Trains per Episode":  number of world instances for training to generate
        in sequence each episode
    "Tests per Episode":  number of world instances for testing to generate
        in sequence each episode
    "Number of Episodes": number of episodes (i.e. generations) in the trial
    "Episode Index": the index of the current episode in the trial
    "Mode": the current simulation mode which can be set to "Train" or "Test"
        Training mode runs before testing mode
    "World Index": the index of the current world instance in the current mode
        and episode
    "Step Index": the index of the current time step for the current world 
        instance
Warning: Use caution when manually reseting these values within the simulation.

"""


class parameters():

    data = {
        "Number of Agents": 30,
        "Number of POIs": 8,
        "Minimum Distance": 1.0,
        "Steps": 100,
        "Trains per Episode": 50,
        "Tests per Episode": 1,
        "Number of Episodes": 5000,
        "World Width": 50,
        "World Length": 50,
        "Poi Static Values": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        "Poi Relative Static Positions": np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.5],
        [0.5, 1.0],
        [0.0, 5.0],
        [0.5, 0.0]
        ]),
        "Agent Initialization Size": 0.1,
        "Coupling": 6,
        "Observation Radius": 4.0,
        "Reward Function": calc_global_reward,
        "Evaluation Function": calc_global_reward,
        "Mod Name": "global",
        "Specifics Name": "30Agents_8Poi_6Coup_Long_Comparison"
    }
