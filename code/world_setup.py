import numpy as np

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

    data['Agent Positions BluePrint'] = np.ones((number_agents,2)) * 0.5 * [world_width, world_length]
    angles = np.random.uniform(-np.pi, np.pi, number_agents)
    data['Agent Orientations BluePrint'] = np.vstack((np.cos(angles), np.sin(angles))).T
    data['Poi Positions BluePrint'] = data['Poi Relative Static Positions'] * [world_width, world_length]
    data['Poi Values BluePrint'] =  data['Poi Static Values'].copy()


def assign_random_policies(data):
    number_agents = data['Number of Agents']
    agent_populations = data['Agent Populations']
    agent_policies = [None] * number_agents
    for agentIndex in range(number_agents):
        agent_policies[agentIndex] = np.random.choice(agent_populations[agentIndex])
    data["Agent Policies"] = agent_policies