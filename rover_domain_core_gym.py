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

        self.trialBeginFuncCol = []
        self.trainBeginFuncCol = []
        self.worldTrainBeginFuncCol = []
        self.worldTrainStepFuncCol = []
        self.worldTrainEndFuncCol = []
        self.trainEndFuncCol = []
        self.testBeginFuncCol = []
        self.worldTestBeginFuncCol = []
        self.worldTestStepFuncCol = []
        self.worldTestEndFuncCol = []
        self.testEndFuncCol = []
        self.trialEndFuncCol = []

        self.trainBeginFuncCol.append(blueprintStatic)
        self.trainBeginFuncCol.append(blueprintAgentInitSize)
        self.worldTrainBeginFuncCol.append(initWorld)
        self.testBeginFuncCol.append(blueprintStatic)
        self.testBeginFuncCol.append(blueprintAgentInitSize)
        self.worldTestBeginFuncCol.append(initWorld)
    
    
        # Add Rover Domain Dynamic Functionality
        """
        step() parameter [action] (2d numpy array with double precision):
            Actions for all rovers before clipping -1 to 1 defined by 
            do_agent_move.
            Dimensions are agentCount by 2.
            
        step()/reset() return [observation] (2d numpy array with double
            precision): Observation for all agents defined by data["Observation 
            Function"].
            Dimensions are agentCount by 8.
            
        For gym compatibility, p.data["Observation Function"] is
        called automatically by this object, no need to call it in a 
        function collection
        """
        p.data["Observation Function"] = get_agent_state
        self.worldTrainStepFuncCol.append(do_agent_move)
        self.worldTestStepFuncCol.append(do_agent_move)

        
            
        # Add Agent Training Reward and Evaluation Functionality
        """
        Training Mode:
        step() return [reward] (1d numpy array with double precision): Reward 
            defined by data["Reward Function"]
            Length is agentCount.
            
        Testing Mode:
        step() return [reward] (double): Performance defined by 
            data["Evaluation Function"]
        """
        self.worldTrainBeginFuncCol.append(create_trajectory_histories)
        self.worldTrainStepFuncCol.append(update_trajectory_histories)
        self.worldTestBeginFuncCol.append(create_trajectory_histories)
        self.worldTestStepFuncCol.append(update_trajectory_histories)
        
        self.worldTrainBeginFuncCol.append(
            lambda data: data.update({"Gym Reward": np.zeros(data['Number of Agents'])})
        )
        self.worldTestBeginFuncCol.append(
            lambda data: data.update({"Gym Reward": 0})
        )
        self.worldTrainEndFuncCol.append(
            lambda data: data["Reward Function"](data)
        )
        self.worldTrainEndFuncCol.append(
            lambda data: data.update({"Gym Reward": data["Agent Rewards"]})
        )
        self.worldTestEndFuncCol.append(
            lambda data: data["Evaluation Function"](data)
        )
        self.worldTestEndFuncCol.append(
            lambda data: data.update({"Gym Reward": data["Global Reward"]}) 
        )    
        
        # Setup world for first time
        self.reset(newMode = "Train", fullyResetting = True)
        
    def step(self, action):
        """
        Proceed 1 time step in world if world is not done
        
        Args:
        action: see rover domain dynamic functionality comments in __init__()
        
        Returns:
        observation: see rover domain dynamic functionality comments in 
            __init__()
        reward: see agent training reward functionality comments for 
            data["Mode"] == "Test" and performance recording functionality 
            comment for data["Mode"] == "Test"
        done (boolean): Describes with the world is done or not
        info (dictionary): The state of the simulation as a dictionary of data
        
        """
        # Store Action for other functions to use
        p.data["Agent Actions"] = action

        
        # If not done, do step functionality
        if p.data["Step Index"] < p.data["Steps"]:
            
            # Do Step Functionality
            p.data["Agent Actions"] = action
            if p.data["Mode"] == "Train":
                for func in self.worldTrainStepFuncCol:
                    func(p.data)
            elif p.data["Mode"] == "Test":
                for func in self.worldTestStepFuncCol:
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
                    for func in self.worldTrainEndFuncCol:
                        func(p.data)
                elif p.data["Mode"] == "Test":
                    for func in self.worldTestEndFuncCol:
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
                
        return p.data["Agent Observations"], p.data["Gym Reward"], \
            done, p.data
        
    def reset(self, newMode = None, fullyResetting = False):
        """
        Reset the world 
            
        Args:
        mode (None, String): Set to "Train" to enable functions associated with 
            training mode. Set to "Test" to enable functions associated with 
            testing mode instead. If None, does not change current simulation 
            mode.
        fullyResetting (boolean): If true, do addition functions 
            (self.trainBeginFuncCol) when setting up world. Typically used for
            resetting the world for a different episode and/or different
            training/testing simulation mode.
            
        Returns:
        observation: see rover domain dynamic functionality comments in 
            __init__()
        """
        # Zero step index for future step() calls
        p.data["Step Index"] = 0
        
        # Set mode if not None
        if newMode != None:
            p.data["Mode"] = newMode
        
        # Execute setting functionality
        if p.data["Mode"] == "Train":
            if fullyResetting:
                for func in self.trainBeginFuncCol:
                    func(p.data)
            for func in self.worldTrainBeginFuncCol:
                func(p.data)
        elif p.data["Mode"] == "Test":
            if fullyResetting:
                for func in self.testBeginFuncCol:
                    func(p.data)
            for func in self.worldTestBeginFuncCol:
                func(p.data)
        else:
            raise Exception('data["Mode"] should be set to "Train" or "Test"')
        
        # Observe state, store result in p.data
        p.data["Observation Function"](p.data)
        
        return p.data["Agent Observations"]
        
def assign(data, key, value):
    data[key] = value