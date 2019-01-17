# Dependencies: numpy, cython 

from parameters import Parameters as p
import numpy as np
import pyximport; pyximport.install() # For cython(pyx) code
from code.agent import * # Rover Domain Dynamic
from code.trajectory_history import create_trajectory_histories, save_trajectory_histories, update_trajectory_histories


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

# World Setup ----------------------------------------------------------------------------------------

def blueprint_agent(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']

    # Initialize all agents in the np.randomly in world
    # Agent positions are a numpy array of size m x n, m = n_agents, n = 2
    data['Agent Positions BluePrint'] = np.random.rand(number_agents, 2) * [world_width, world_length]
    rover_angles = np.random.uniform(-np.pi, np.pi, number_agents) # Rover orientations
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(rover_angles), np.sin(rover_angles))).T


def blueprint_agent_init_size(data):
    number_agents = data['Number of Agents']
    world_width = data['World Width']
    world_length = data['World Length']
    agent_init_size = data["Agent Initialization Size"]

    world_size = np.array([world_width, world_length])

    # Initialize all agents in the np.randomly in world
    agent_positions = np.random.rand(number_agents, 2) * world_size
    agent_positions *= agent_init_size
    agent_positions += 0.5 * (1 - agent_init_size) * world_size
    data['Agent Positions BluePrint'] = agent_positions
    rover_angles = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(rover_angles), np.sin(rover_angles))).T


def blueprint_poi(data):
    number_pois = data['Number of POIs']
    world_width = data['World Width']
    world_length = data['World Length']

    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(number_pois, 2) * [world_width, world_length]
    data['Poi Values BluePrint'] = np.arange(number_pois) + 1.0


def init_world(data):
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Agent Orientations'] = data['Agent Orientations BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
    data['Poi Values'] = data['Poi Values BluePrint'].copy()


def blueprint_static(data):
    number_agents = data['Number of Agents']
    number_pois = data['Number of POIs']
    world_width = data['World Width']
    world_length = data['World Length']

    data['Agent Positions BluePrint'] = np.ones((number_agents, 2)) * 0.5 * [world_width, world_length]
    angles = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angles), np.sin(angles))).T
    data['Poi Positions BluePrint'] = data['Poi Relative Static Positions'] * [world_width, world_length]
    data['Poi Values BluePrint'] =  data['Poi Static Values'].copy()


def assign_random_policies(data):
    number_agents = data['Number of Agents']
    agent_populations = data['Agent Populations']
    agent_policies = [None] * number_agents
    for agent_id in range(number_agents):
        agent_policies[agent_id] = np.random.choice(agent_populations[agent_id])
    data["Agent Policies"] = agent_policies


class rover_domain_core_gym():

    def __init__(self):

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
        self.agent_setup_train.append(blueprint_poi)
        self.agent_setup_train.append(blueprint_agent)
        self.world_setup_train.append(init_world)
        self.agent_setup_test.append(blueprint_poi)
        self.agent_setup_test.append(blueprint_agent)
        self.world_setup_test.append(init_world)

        # Add Rover Domain Dynamic Functionality
        p.data["Observation Function"] = get_agent_state
        self.world_update_functions_train.append(do_agent_move)
        self.world_update_functions_test.append(do_agent_move)
            
        # Add Agent Training Reward and Evaluation Functionality
        # Setup functions
        self.world_setup_train.append(create_trajectory_histories)
        self.world_setup_test.append(create_trajectory_histories)
        self.world_setup_train.append(
            lambda data: data.update({"Gym Reward": np.zeros(data['Number of Agents'])})
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

        
    def step(self, action): # Agents do actions for one time step

        # Store Action for other functions to use
        p.data["Agent Actions"] = action

        
        # If not done, do step functionality
        if p.data["Step Index"] < p.data["Steps"]:
            
            # Do Step Functionality
            p.data["Agent Actions"] = action
            if p.data["Mode"] == "Train":
                for func in self.world_update_functions_train:
                    func(p.data)
            elif p.data["Mode"] == "Test":
                for func in self.world_update_functions_test:
                    func(p.data)
            else:
                raise Exception(
                    'data["Mode"] should be set to "Train" or "Test"'
                )
            
            # Increment step index for future step() calls
            p.data["Step Index"] += 1
            
            # Check is world is done; if so, do ending functions
            if p.data["Step Index"] >= p.data["Steps"]:
                if p.data["Mode"] == "Train":
                    for func in self.evaluate_world_results_train:
                        func(p.data)
                elif p.data["Mode"] == "Test":
                    for func in self.evaluate_world_results_test:
                        func(p.data)
                else:
                    raise Exception(
                        'data["Mode"] should be set to "Train" or "Test"'
                    )
                    
            # Observe state, store result in p.data
            p.data["Observation Function"](p.data)
        
        # Check if simulation is done
        done = False
        if p.data["Step Index"] >= p.data["Steps"]:
            done = True
                
        return p.data["Agent Observations"], p.data["Gym Reward"], done, p.data

        
    def reset(self, new_mode = None, fully_resetting = False):

        # Zero step index for future step() calls
        p.data["Step Index"] = 0
        
        # Set mode if not None
        if new_mode != None:
            p.data["Mode"] = new_mode
        
        # Execute setting functionality
        if p.data["Mode"] == "Train":
            if fully_resetting:
                for func in self.agent_setup_train: #Go through list of functions in agent_setup_train
                    func(p.data)
            for func in self.world_setup_train: #Go through list of functions in world_setup_train
                func(p.data)

        elif p.data["Mode"] == "Test":
            if fully_resetting:
                for func in self.agent_setup_test:
                    func(p.data)
            for func in self.world_setup_test:
                func(p.data)
        else:
            raise Exception('data["Mode"] should be set to "Train" or "Test"')
        
        # Observe state, store result in p.data
        p.data["Observation Function"](p.data)
        
        #return p.data["Agent Observations"]
        
def assign(data, key, value):
    data[key] = value