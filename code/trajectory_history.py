import csv
import os
import errno
import numpy as np
from parameters import Parameters as p


def create_trajectory_histories(data):
    # create a history of positions for each agents in order to evalutate reward at the end
    number_agents = p.number_of_agents
    history_step_count = p.total_steps + 1
    agent_position_history = np.zeros((number_agents, history_step_count, 2))
    agent_positions = data["Agent Positions"]

    for rover_id in range(p.number_of_agents):
        agent_position_history[rover_id, 0] = agent_positions[rover_id]

    data["Agent Position History"] = agent_position_history
    
    
def update_trajectory_histories(data):
    step_index = data["Step Index"]
    agent_position_history = data["Agent Position History"]
    agent_positions = data["Agent Positions"]
    
    for rover_id in range(p.number_of_agents):
        agent_position_history[rover_id, step_index + 1] = agent_positions[rover_id]
        
    data["Agent Position History"] = agent_position_history
    
def save_trajectory_histories(data):
    save_file_name = data["Trajectory Save File Name"]
    number_agents = p.number_of_agents
    number_pois = p.number_of_pois
    agent_position_history = data["Agent Position History"]
    poi_positions = data["Poi Positions"]
    
    if not os.path.exists(os.path.dirname(save_file_name)):
        try:
            os.makedirs(os.path.dirname(save_file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(save_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for agent_id in range(number_agents):
            writer.writerow(["Agent %d Position 0" % agent_id] + [pos[0] for pos in agent_position_history[agent_id, :, :]])
            writer.writerow(["Agent %d Position 1" % agent_id] + [pos[1] for pos in agent_position_history[agent_id, :, :]])

        for poi_id in range(number_pois):
            writer.writerow(["Poi %d Position 0" % poi_id] + [poi_positions[poi_id, 0]])
            writer.writerow(["Poi %d Position 1" % poi_id] + [poi_positions[poi_id, 1]])
