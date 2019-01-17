import pyximport; pyximport.install()  # For cython(pyx) code

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

    # Run Parameters
    stat_runs = 1
    generations = 20
    tests_per_episode = 1
    number_of_episodes = 1

    # Domain parameters
    number_of_agents = 12
    number_of_pois = 10
    min_distance = 1
    total_steps = 30
    world_width = 20
    world_length = 30
    coupling = 3
    activation_dist = 4

    # Neural network parameters
    number_of_inputs = 8
    number_of_nodes = 10
    number_of_outputs = 2

    # CCEA parameters
    mutation_rate = 0.9
    epsilon = 0.1
    population_size = 10
