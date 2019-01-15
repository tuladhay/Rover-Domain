# Dependencies: numpy, cython 

#from core import simulation_core
from parameters import parameters as p
import pyximport; pyximport.install() # For cython(pyx) code
from code.world_setup import * # Rover Domain Construction
from code.agent_domain import * # Rover Domain Dynamic
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
        self.trainEndFuncCol = []
        self.evaluate_world_results_test = []
        self.testEndFuncCol = []

        # Other
        self.trialBeginFuncCol = []
        self.trialEndFuncCol = []

        # Add setup functions to function call (THESE MUST BE ADDED FIRST)
        self.agent_setup_train.append(blueprint_static)
        self.agent_setup_train.append(blueprint_agent_init_size)
        self.world_setup_train.append(init_world)
        self.agent_setup_test.append(blueprint_static)
        self.agent_setup_test.append(blueprint_agent_init_size)
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
        
        return p.data["Agent Observations"]
        
def assign(data, key, value):
    data[key] = value