# from parameters import parameters as p
#
# """
# SimulationCore class is the backbone of any arbitrary domain.
# By default, SimulationCore does not provide any domain-specific functionality.
# To provide functionality, modify the following attributes:
#
# Note: Each function (or callable class) must take in the dictionary data as its
#     first and only required parameter.
#
# trialBeginFuncCol:
# An ordered collection of functions; each function is executed in order at the
#     beginning of the trial. These functions should set the entire simulation
#     trial.
#
# trainBeginFuncCol:
# An ordered collection of functions; each function is executed in order at the
#     beginning of the training mode of the current episode. These functions
#     should set the episode for training. Train mode runs before test mode.
#
# worldTrainBeginFuncCol:
# An ordered collection of functions; each function is executed in order at the
#     beginning of the world instance of the current episode when in training
#     mode. These functions should set the world.
#
# worldTrainStepFuncCol:
# An ordered collection of functions; each function is executed in order
#     for each step in the current world instance when in training
#     mode.
#
# self.worldTrainEndFuncCol:
# An ordered collection of functions; each function is executed in order
#     at the end of the world instance of the current episode when
#     in training mode.
#
# self.trainEndFuncCol:
# An ordered collection of functions; each function is executed in order
#     at the end of the training mode of the current episode.
#
# testBeginFuncCol:
# An ordered collection of functions; each function is executed in order at the
#     beginning of the testing mode of the current episode. These functions
#     should set the episode for testing. Train mode runs before test mode.
#
# worldTrainBeginFuncCol:
# An ordered collection of functions; each function is executed in order at the
#     beginning of the world instance of the current episode when in testing
#     mode. These functions should set the world.
#
# worldTrainStepFuncCol:
# An ordered collection of functions; each function is executed in order
#     for each step in the current world instance when in testing data
#     mode.
#
# self.worldTrainEndFuncCol:
# An ordered collection of functions; each function is executed in order
#     at the end of the world instance of the current episode when
#     in testing mode.
#
# self.testEndFuncCol:
# An ordered collection of functions; each function is executed in order
#     at the end of the testing mode of the current episode.
#
# trialEndFuncCol:
# An ordered collection of functions; each function is executed in order at the
#     end of the trial. Some of these functions should save important trial
#     information.
#
# Note: Each function must take in the dictionary data as its first and only required
#     parameter.
# """
# class simulation_core:
#     def __init__(self):
#         """
#         Run function executes a new simulation trial by runnning prescribed
#             functions at prescribed times
#
#         Args:
#
#         Returns:
#             None
#         """
#
#         self.trialBeginFuncCol = []
#         self.trainBeginFuncCol = []
#         self.worldTrainBeginFuncCol = []
#         self.worldTrainStepFuncCol = []
#         self.worldTrainEndFuncCol = []
#         self.trainEndFuncCol = []
#         self.testBeginFuncCol = []
#         self.worldTestBeginFuncCol = []
#         self.worldTestStepFuncCol = []
#         self.worldTestEndFuncCol = []
#         self.testEndFuncCol = []
#         self.trialEndFuncCol = []
#
#     # def run(self):
#     #     """
#     #     Run function executes the simulation but runnning prescribed functions
#     #         at prescribed times
#     #
#     #     Args:
#     #
#     #     Returns:
#     #         None
#     #     """
#     #
#     #     # Do Trial Begin Functions
#     #     for func in self.trialBeginFuncCol:
#     #         func(p.data)
#     #
#     #     # Do Each Episode
#     #     for episodeIndex in range(p.data["Number of Episodes"]):
#     #         p.data["Episode Index"] = episodeIndex
#     #
#     #         # Do Begin Training Functions
#     #         for func in self.trainBeginFuncCol:
#     #             func(p.data)
#     #
#     #         # Repeat running world (with new teams) until repeat is set to false
#     #         for worldIndex in range(p.data["Trains per Episode"]):
#     #             p.data["Mode"] = "Train"
#     #             p.data["World Index"] = worldIndex
#     #             p.data["Step Index"] = None
#     #
#     #             # Do world begin (setup) functions
#     #             for func in self.worldTrainBeginFuncCol:
#     #                 func(p.data)
#     #
#     #             # Do world end functions
#     #             for stepIndex in range(p.data["Steps"]):
#     #                 p.data["Step Index"] = stepIndex
#     #                 for func in self.worldTrainStepFuncCol:
#     #                     func(p.data)
#     #
#     #             # Do world
#     #             for func in self.worldTrainEndFuncCol:
#     #                 func(p.data)
#     #
#     #         # Do End Training Functions
#     #         for func in self.trainEndFuncCol:
#     #             func(p.data)
#     #
#     #         # Do Begin Testing Functions
#     #         for func in self.testBeginFuncCol:
#     #             func(p.data)
#     #
#     #         # Repeat running world (with new teams) until repeat is set to false
#     #         for worldIndex in range(p.data["Tests per Episode"]):
#     #             p.data["Mode"] = "Test"
#     #             p.data["World Index"] = worldIndex
#     #             p.data["Step Index"] = None
#     #
#     #
#     #             # Do world begin (setup) functions
#     #             for func in self.worldTestBeginFuncCol:
#     #                 func(p.data)
#     #
#     #             # Do world end functions
#     #             for stepIndex in range(p.data["Steps"]):
#     #                 p.data["Step Index"] = stepIndex
#     #                 for func in self.worldTestStepFuncCol:
#     #                     func(p.data)
#     #
#     #             # Do world
#     #             for func in self.worldTestEndFuncCol:
#     #                 func(p.data)
#     #
#     #         # Do End Testing Functions
#     #         for func in self.testEndFuncCol:
#     #             func(p.data)
#     #
#     #     # Do Trial End Functions
#     #     for func in self.trialEndFuncCol:
#     #         func(p.data)
