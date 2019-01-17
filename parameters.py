"""
This file contains all constant test parameters which may be altered from this single
location for convenience.
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
    min_distance = 1  # Minimum distance which may appear in the denominator of credit eval functions
    total_steps = 30  # Number of steps rovers take during each run of the world
    world_width = 30
    world_length = 30
    coupling = 3  # Number of rovers required to view a POI for credit
    activation_dist = 4  # Minimum distance rovers must be to observe POIs

    # Neural network parameters
    number_of_inputs = 8
    number_of_nodes = 10
    number_of_outputs = 2

    # CCEA parameters
    mutation_rate = 0.9
    epsilon = 0.1
    population_size = 10
