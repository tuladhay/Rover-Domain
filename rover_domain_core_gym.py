# Dependencies: numpy, cython 

import pyximport; pyximport.install() # For cython(pyx) code
from parameters import Parameters as p
from code.agent import * # Rover Domain Dynamic
from code.trajectory_history import create_trajectory_histories, update_trajectory_histories
import numpy as np
from code.reward import calc_global_reward


"""
Provides Open AI gym wrapper for rover domain selfulation core with some extra
    gym-specific functionality. This is the gym equivalent to 'getSim()' in 
    the specific.py file.
    
    Get a default rover domain simulation with some default functionality.
    Users are encouraged to modify this function and save copies of it for
     each trial to use as a parameter reference.
    
Set data["Reward Function"] to define the reward function callback
Set data["Evaluation Function"] to define the evaluation function callback
Set data["Observation Function"] to define the observation funciton callback

Note: step function returns result of either the reward or evaluation function 
    depending mode ("Train" vs "Test" respectively)

RoverDomainCoreGym should be mods 
"""

"""
data:
A dictionary shared amongst all functions in the simulation.
User may add any property they may want to have shared by all provided functions
SimulationCore provides and manages the following keys during run() execution:
    "Total Steps": duration of world measured in time steps,
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

# World Setup ----------------------------------------------------------------------------------------
# Randomly initalize agent positions on map
def blueprint_agent(data):
    number_agents = p.number_of_agents
    world_width = p.world_width
    world_length = p.world_length

    # Agent positions are a numpy array of size m x n, m = n_agents, n = 2
    data['Agent Positions BluePrint'] = np.random.rand(number_agents, 2) * [world_width, world_length]
    rover_angles = np.random.uniform(-np.pi, np.pi, number_agents) # Rover orientations
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(rover_angles), np.sin(rover_angles))).T

# Randomly initialize POI positions on map
def blueprint_poi(data):
    number_pois = p.number_of_pois
    world_width = p.world_width
    world_length = p.world_length

    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(number_pois, 2) * [world_width, world_length]
    data['Poi Values BluePrint'] = np.arange(number_pois) + 1.0


def init_world(data):
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Agent Orientations'] = data['Agent Orientations BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
    data['Poi Values'] = data['Poi Values BluePrint'].copy()


# Agent positions are statically set on the mapblueprint_static
def blueprint_static(data):
    number_agents = p.number_of_agents
    number_pois = p.number_of_pois
    world_width = p.world_width
    world_length = p.world_length

    data['Agent Positions BluePrint'] = np.ones((number_agents, 2)) * 0.5 * [world_width, world_length]
    angles = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angles), np.sin(angles))).T
    data['Poi Positions BluePrint'] = data['Poi Relative Static Positions'] * [world_width, world_length]
    data['Poi Values BluePrint'] = data['Poi Static Values'].copy()


class RoverDomainCore:

    def __init__(self):

        self.data = {
            # Agent values
            "Agent Positions": np.zeros((p.number_of_agents, 2)),
            "Agent Orientations": np.zeros((p.number_of_agents, 2)),
            "Agent Positions BluePrint": np.zeros((p.number_of_agents, 2)),
            "Agent Orientations BluePrint": np.zeros((p.number_of_agents, 2)),
            "Agent Observations": np.zeros((p.number_of_agents, 8)),
            "Agent Position History": np.zeros((p.number_of_agents, p.total_steps, 2)),
            "Agent Orientation History": np.zeros((p.number_of_agents, p.total_steps, 2)),

            # POI values
            "Poi Positions": np.zeros((p.number_of_pois, 2)),
            "Poi Values": np.zeros((p.number_of_pois, 1)),
            "Poi Values BluePrint": np.zeros((p.number_of_pois, 1)),
            "Poi Positions BluePrint": np.zeros((p.number_of_pois, 2)),

            # Domain values
            "Reward Function": calc_global_reward,
            "Evaluation Function": calc_global_reward,
            "Total Steps": p.total_steps,
            "Mod Name": "global",
            "Step Index": 0,
            "Global Reward": 0.0,

            # Data file values
            "Specifics Name": "12Agents_10Poi_3Coup_Long_Comparison",  # Name of save file for data
            "Performance Save File Name": "Test_Data",
            "Trajectory Save File Name": "Trajectory_Data",
            "Pickle Save File Name": "pickle_data"
        }

        # Setup functions:
        self.agent_setup_train = []
        self.world_setup_train = []
        self.agent_setup_test = []
        self.world_setup_test = []

        # Update functions:
        self.world_update_functions_train = []
        self.world_update_functions_test = []

        # Results evaluation functions:
        self.evaluate_world_results_train = []
        self.train_end_functions = []
        self.evaluate_world_results_test = []
        self.test_end_functions = []

        # Other
        self.trial_begin_functions = []
        self.trial_end_functions = []

        # Add setup functions to function call (THESE MUST BE ADDED FIRST)
        self.agent_setup_train.append(blueprint_poi)  # Initialize POI positions and values
        self.agent_setup_train.append(blueprint_agent)  # Initialize agent positions and orientations
        self.world_setup_train.append(init_world)  # Set arrays equal to blueprints
        self.agent_setup_test.append(blueprint_poi)
        self.agent_setup_test.append(blueprint_agent)
        self.world_setup_test.append(init_world)

        # Add Rover Domain Dynamic Functionality
        self.data["Observation Function"] = get_agent_state
        self.world_update_functions_train.append(do_agent_move)
        self.world_update_functions_test.append(do_agent_move)
            
        # Add Agent Training Reward and Evaluation Functionality
        # Setup functions
        self.world_setup_train.append(create_trajectory_histories)
        self.world_setup_test.append(create_trajectory_histories)
        self.world_setup_train.append(
            lambda data: data.update({"Gym Reward": np.zeros(p.number_of_agents)})
        )
        self.world_setup_test.append(
            lambda data: data.update({"Gym Reward": 0})
        )

        # Update functions
        self.world_update_functions_train.append(update_trajectory_histories)
        self.world_update_functions_test.append(update_trajectory_histories)

        # Results evaluation functions
        self.evaluate_world_results_train.append(
            lambda data: data["Reward Function"](data)
        )
        self.evaluate_world_results_train.append(
            lambda data: data.update({"Gym Reward": data["Agent Rewards"]})
        )
        self.evaluate_world_results_test.append(
            lambda data: data["Evaluation Function"](data)
        )
        self.evaluate_world_results_test.append(
            lambda data: data.update({"Gym Reward": data["Global Reward"]}) 
        )    


        # Setup world for first time
        self.reset(new_mode = "Train", fully_resetting = True)

        
    def step(self): # Agents do actions for one time step
        
        # If not done, do step functionality
        if self.data["Step Index"] < self.data["Total Steps"]:
            
            # Do Step Functionality
            if self.data["Mode"] == "Train":
                for func in self.world_update_functions_train:  # do_agent_move
                    func(self.data)
            elif self.data["Mode"] == "Test":
                for func in self.world_update_functions_test:  # do_agent_move
                    func(self.data)
            else:
                raise Exception(
                    'data["Mode"] should be set to "Train" or "Test"'
                )
            
            # Increment step index for future step() calls
            self.data["Step Index"] += 1
            
            # Check is world is done; if so, do ending functions
            if self.data["Step Index"] >= self.data["Total Steps"]:
                if self.data["Mode"] == "Train":
                    for func in self.evaluate_world_results_train:
                        func(self.data)
                elif self.data["Mode"] == "Test":
                    for func in self.evaluate_world_results_test:
                        func(self.data)
                else:
                    raise Exception(
                        'data["Mode"] should be set to "Train" or "Test"'
                    )
                    
            # Observe state, store result in self.data
            self.data["Observation Function"](self.data)
        
        # Check if simulation is done
        done = False
        if self.data["Step Index"] >= self.data["Total Steps"]:
            done = True
                
        return self.data["Agent Observations"], self.data["Gym Reward"], done, self.data  # Gym Reward is Agent Rewards

        
    def reset(self, new_mode = None, fully_resetting = False):

        # Zero step index for future step() calls
        self.data["Step Index"] = 0
        
        # Set mode if not None
        if new_mode != None:
            self.data["Mode"] = new_mode
        
        # Execute setting functionality
        if self.data["Mode"] == "Train":
            if fully_resetting:
                for func in self.agent_setup_train: #Go through list of functions in agent_setup_train
                    func(self.data)
            for func in self.world_setup_train: #Go through list of functions in world_setup_train
                func(self.data)

        elif self.data["Mode"] == "Test":
            if fully_resetting:
                for func in self.agent_setup_test:
                    func(self.data)
            for func in self.world_setup_test:
                func(self.data)
        else:
            raise Exception('data["Mode"] should be set to "Train" or "Test"')
        
        # Observe state, store result in self.data (Get initial state)
        self.data["Observation Function"](self.data)
        
def assign(data, key, value):
    data[key] = value
