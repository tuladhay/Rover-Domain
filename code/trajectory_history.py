import csv
import os
import errno
import numpy as np


def create_trajectory_histories(data):
    # create a history of positions for each agents in order to evalutate reward at the end
    number_agents = data['Number of Agents']
    history_step_count = data["Steps"] + 1
    agent_position_history = np.zeros((history_step_count, number_agents, 2))
    agent_orientation_history = np.zeros((history_step_count, number_agents, 2))
    agent_positions = data["Agent Positions"]
    agent_orientations = data["Agent Orientations"]

    agent_position_history[0] = agent_positions
    agent_orientation_history[0] = agent_orientations
    
    
    data["Agent Position History"] = agent_position_history
    data["Agent Orientation History"] = agent_orientation_history
    
    
def update_trajectory_histories(data):
    number_agents = data['Number of Agents']
    step_index = data["Step Index"]
    history_step_count = data["Steps"] + 1
    agent_position_history = data["Agent Position History"]
    agent_orientation_history = data["Agent Orientation History"]
    agent_positions = data["Agent Positions"]
    agent_orientations = data["Agent Orientations"]
    
    
    agent_position_history[step_index + 1] = agent_positions
    agent_orientation_history[step_index + 1] = agent_orientations
        
    data["Agent Position History"] = agent_position_history
    data["Agent Orientation History"] = agent_orientation_history
    
def save_trajectory_histories(data):
    save_file_name = data["Trajectory Save File Name"]
    number_agents = data['Number of Agents']
    number_pois = data["Number of POIs"]
    history_step_count = data["Steps"] + 1
    agent_position_history = data["Agent Position History"]
    agent_orientation_history = data["Agent Orientation History"]
    poi_positions = data["Poi Positions"]
    
    # if not os.path.exists(os.path.dirname(save_file_name)):
    #     try:
    #         os.makedirs(os.path.dirname(save_file_name))
    #     except OSError as exc: # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    #
    # with open(save_file_name, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #
    #     for agent_id in range(number_agents):
    #         writer.writerow(["Agent %d Position 0" % agent_id] + [pos[0] for pos in agent_position_history[:, agent_id, :]])
    #         writer.writerow(["Agent %d Position 1" % agent_id] + [pos[1] for pos in agent_position_history[:, agent_id, :]])
    #         writer.writerow(["Agent %d Orientation 0" % agent_id] + [ori[0] for ori in agent_orientation_history[:, agent_id, :]])
    #         writer.writerow(["Agent %d Orientation 1" % agent_id] + [ori[1] for ori in agent_orientation_history[:, agent_id, :]])
    #
    #     for poi_id in range(number_pois):
    #         writer.writerow(["Poi %d Position 0" % poi_id] + [poi_positions[poi_id, 0]])
    #         writer.writerow(["Poi %d Position 1" % poi_id] + [poi_positions[poi_id, 1]])
