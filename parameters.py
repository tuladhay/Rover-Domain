import pyximport; pyximport.install()  # For cython(pyx) code
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


class Parameters:

    data = {
        "Number of Agents": 4,
        "Number of POIs": 10,
        "Minimum Distance": 1.0,
        "Steps": 10,
        "Generations per Episode": 10,
        "Tests per Episode": 1,
        "Number of Episodes": 1,
        "World Width": 30,  # X-Dimension
        "World Length": 30,  # Y-Dimension
        "Number of Inputs": 1,  # NN inputs
        "Number of Nodes": 3,  # NN hidden nodes
        "Number of Outputs": 2,  # NN outputs
        "Agent Initialization Size": 0.1,
        "Coupling": 3,  # How many rovers are required to observe a POI
        "Activation Radius": 4.0,  # Minimum distance at which a POI may be observed for credit
        "Reward Function": calc_global_reward,
        "Evaluation Function": calc_global_reward,
        "Mod Name": "global",
        "Specifics Name": "12Agents_10Poi_3Coup_Long_Comparison",
        "Mutation Rate": 0.9,  # How likely a given policy is to be mutated
        "Population Size": 2,  # Number of policies in each population
        "Epsilon": 0.1  # Epsilon constant for e-greedy selection in CCEA
    }
