# Dependencies: numpy, cython 

import pyximport; pyximport.install() # For cython(pyx) code
from parameters import Parameters as p
from code.agent import get_state_vec, do_agent_move, get_agent_actions  # Rover functions
from code.trajectory_history import create_trajectory_histories, update_trajectory_histories
import numpy as np
from code.reward import calc_global_reward

# World Setup ----------------------------------------------------------------------------------------
# Randomly initalize agent positions on map
def init_agents_random(data):
    number_agents = p.number_of_agents
    x_dim = p.world_width
    y_dim = p.world_length

    # Agent positions are a numpy array of size m x n, m = n_agents, n = 2
    data['Agent Positions BluePrint'] = np.random.rand(number_agents, 2) * [x_dim, y_dim]

# Initialze agents statically
def init_agents_static(data):
    number_agents = p.number_of_agents
    x_dim = p.world_width
    y_dim = p.world_length

    # Agent positions are a numpy array of size m x n, m = n_agents, n = 2
    data['Agent Positions BluePrint'] = np.ones((number_agents, 2)) * (x_dim/2)

# Randomly initialize POI positions on map
def init_pois_random(data):
    number_pois = p.number_of_pois
    x_dim = p.world_width
    y_dim = p.world_length

    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(number_pois, 2) * [x_dim, y_dim]
    data['Poi Values BluePrint'] = np.ones(number_pois)
    for i in range(number_pois):
        data['Poi Values BluePrint'][i] *= np.random.randint(1, 10)

# Initialize POIs statically
def init_pois_static(data):
    number_pois = p.number_of_pois
    x_dim = p.world_width
    y_dim = p.world_length

    # Initialize all Pois np.randomly
    data['Poi Positions BluePrint'] = np.random.rand(number_pois, 2) * [x_dim, y_dim]
    data['Poi Values BluePrint'] = np.ones(number_pois) * 5


def init_world(data):
    data['Agent Positions'] = data['Agent Positions BluePrint'].copy()
    data['Poi Positions'] = data['Poi Positions BluePrint'].copy()
    data['Poi Values'] = data['Poi Values BluePrint'].copy()
    # for rov_id in range(p.number_of_agents):
    #     print('Rover: ', data['Agent Positions'][rov_id, 0], data['Agent Positions'][rov_id, 1])
    # for poi_id in range(p.number_of_pois):
    #     print('POI: ', data['Poi Positions'][poi_id, 0], data['Poi Positions'][poi_id, 1])


# Agent positions are statically set on the mapblueprint_static
def blueprint_static(data):
    number_agents = p.number_of_agents
    number_pois = p.number_of_pois
    world_width = p.world_width
    world_length = p.world_length

    data['Agent Positions BluePrint'] = np.ones((number_agents, 2)) * 0.5 * [world_width, world_length]
    data['Poi Positions BluePrint'] = data['Poi Relative Static Positions'] * [world_width, world_length]
    data['Poi Values BluePrint'] = data['Poi Static Values'].copy()


class RoverDomainCore:
    def __init__(self):
        self.data = {
            # Agent values
            "Agent Positions": np.zeros((p.number_of_agents, 2)),
            "Agent Positions BluePrint": np.zeros((p.number_of_agents, 2)),
            "Agent Observations": np.zeros((p.number_of_agents, 8)),
            "Agent Position History": np.zeros((p.number_of_agents, p.total_steps, 2)),
            "Agent Rewards": np.zeros(p.number_of_agents),

            # POI values
            "Poi Positions": np.zeros((p.number_of_pois, 2)),
            "Poi Values": np.zeros((p.number_of_pois, 1)),
            "Poi Values BluePrint": np.zeros((p.number_of_pois, 1)),
            "Poi Positions BluePrint": np.zeros((p.number_of_pois, 2)),

            # Domain values
            "Reward Function": [],
            "Evaluation Function": calc_global_reward,
            "Mod Name": [],
            "Step Index": 0,
            "Global Reward": 0.0,

            # Data file values
            "Specifics Name": "12Agents_10Poi_3Coup_Long_Comparison",  # Name of save file for data
            "Performance Save File Name": "Test_Data",
            "Trajectory Save File Name": "Trajectory_Data",
            "Pickle Save File Name": "pickle_data",
            "Reward History": [[] for i in range(p.stat_runs)]  # Tracks reward history from best policies
        }

        # Setup world for first time
        self.reset_world(new_mode="Train", fully_resetting = True)

    def step(self):  # Agents do actions for one time step
        # If not done, do step functionality
        if self.data["Step Index"] < p.total_steps:
            do_agent_move(self.data)
            update_trajectory_histories(self.data)
            
            # Increment step index for future step() calls
            self.data["Step Index"] += 1
                    
            # Observe state, store result in self.data
            get_state_vec(self.data)
        
        # Check if simulation is done
        done = False
        if self.data["Step Index"] >= p.total_steps:
            done = True

        # Check if world is done; if so, evaluate performance
        if done == True:
            if self.data["Mode"] == "Train":
                self.data["Reward Function"](self.data)  # Calls on reward function specified by mods
            elif self.data["Mode"] == "Test":
                self.data["Evaluation Function"](self.data)  # Runs function for evaluating best policies
            else:
                raise Exception(
                    'data["Mode"] should be set to "Train " or "Test"'
                )
                
        return done

    def reset_world(self, new_mode=None, fully_resetting=False):
        if fully_resetting == False:
            init_world(self.data)
        else:
            init_agents_random(self.data)
            init_pois_random(self.data)
            init_world(self.data)

        create_trajectory_histories(self.data)
        
        # Set mode if not None
        if new_mode != None:
            self.data["Mode"] = new_mode
        
        # Observe state, store result in self.data (Get initial state)
        get_state_vec(self.data)


def assign(data, key, value):
    data[key] = value
