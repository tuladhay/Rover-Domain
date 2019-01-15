Functions Descriptions and term definitions for items found in rover_domain_core_gym.py

Note: Each function (or callable class) must take in the dictionary data as its
     first and only required parameter.

Training Mode:
        step() return [reward] (1d numpy array with double precision): Reward 
            defined by data["Reward Function"]
            Length is agentCount.
            
Testing Mode:
        step() return [reward] (double): Performance defined by 
            data["Evaluation Function"]

trialBeginFuncCol:
 An ordered collection of functions; each function is executed in order at the
     beginning of the trial. These functions should set the entire simulation
     trial.

trainBeginFuncCol:
 An ordered collection of functions; each function is executed in order at the
     beginning of the training mode of the current episode. These functions
     should set the episode for training. Train mode runs before test mode.

worldTrainBeginFuncCol:
 An ordered collection of functions; each function is executed in order at the
     beginning of the world instance of the current episode when in training
     mode. These functions should set the world.

worldTrainStepFuncCol:
 An ordered collection of functions; each function is executed in order
     for each step in the current world instance when in training
     mode.

self.worldTrainEndFuncCol:
 An ordered collection of functions; each function is executed in order
     at the end of the world instance of the current episode when
     in training mode.

self.trainEndFuncCol:
 An ordered collection of functions; each function is executed in order
     at the end of the training mode of the current episode.

testBeginFuncCol:
 An ordered collection of functions; each function is executed in order at the
     beginning of the testing mode of the current episode. These functions
     should set the episode for testing. Train mode runs before test mode.

worldTrainBeginFuncCol:
 An ordered collection of functions; each function is executed in order at the
     beginning of the world instance of the current episode when in testing
     mode. These functions should set the world.

worldTrainStepFuncCol:
 An ordered collection of functions; each function is executed in order
     for each step in the current world instance when in testing data
     mode.

self.worldTrainEndFuncCol:
 An ordered collection of functions; each function is executed in order
     at the end of the world instance of the current episode when
     in testing mode.

self.testEndFuncCol:
 An ordered collection of functions; each function is executed in order
     at the end of the testing mode of the current episode.

trialEndFuncCol:
 An ordered collection of functions; each function is executed in order at the
     end of the trial. Some of these functions should save important trial
     information.

reset:
        Args:
        mode (None, String): Set to "Train" to enable functions associated with 
            training mode. Set to "Test" to enable functions associated with 
            testing mode instead. If None, does not change current simulation 
            mode.
        fully_resetting (boolean): If true, do addition functions
            (self.trainBeginFuncCol) when setting up world. Typically used for
            resetting the world for a different episode and/or different
            training/testing simulation mode.
            
        Returns:
        observation: see rover domain dynamic functionality comments in 
            __init__()

step:
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
