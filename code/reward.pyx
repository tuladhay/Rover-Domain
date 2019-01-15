import numpy as np
cimport cython

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_global_reward(data):

    # Set parameters from those found in data dictionary
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double min_sqr_dist = data["Minimum Distance"] ** 2
    cdef int total_steps = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double sqr_observation_rad = data["Observation Radius"] ** 2
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count
    cdef double agent_x_dist, agent_y_dist, closest_obs_dist, sqr_distance, closest_rover_sqr_dist
    cdef double inf = float("inf")
    cdef double g_reward = 0.0 # Global reward
    
    # For all POIs
    for poi_id in range(number_pois):
        closest_obs_dist = inf

        # For all timesteps (rover steps)
        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            closest_rover_sqr_dist = inf
            # For all agents

            for agent_id in range(number_agents):
                # Calculate separation distance between poi and agent
                # agent_x_dist is the x-distance and agent_y_dist is the y-distance
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, agent_id, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, agent_id, 1]
                sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist
                
                # Check if agent observes poi, update closest step distance
                if sqr_distance < sqr_observation_rad: # Rover is in observation range
                    observer_count += 1
                    if sqr_distance < closest_rover_sqr_dist:
                        closest_rover_sqr_dist = sqr_distance # Record closest rover to POI thus far
                        
                        
            # update closest distance only if poi is observed    
            if observer_count >= coupling:
                if closest_rover_sqr_dist < closest_obs_dist:
                    closest_obs_dist = closest_rover_sqr_dist
        
        # add to global reward if poi is observed 
        if closest_obs_dist < sqr_observation_rad:
            if closest_obs_dist < min_sqr_dist:
                closest_obs_dist = min_sqr_dist
            g_reward += poi_values[poi_id] / closest_obs_dist
    
    data["Global Reward"] = g_reward
    data["Agent Rewards"] = np.ones(number_agents) * g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_difference_reward(data):
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double min_sqr_dist = data["Minimum Distance"] ** 2
    cdef int total_steps = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double sqr_observation_rad = data["Observation Radius"] ** 2
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id
    cdef double agent_x_dist, agent_y_dist, closest_obs_dist, sqr_distance, closest_rover_sqr_dist
    cdef double inf = float("inf")
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    
    d_reward = np.zeros(number_agents)
    cdef double[:] difference_reward = d_reward
    
    for poi_id in range(number_pois):
        closest_obs_dist = inf

        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            closest_rover_sqr_dist = inf

            for agent_id in range(number_agents):
                # Calculate separation distance between poi and agent
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, agent_id, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, agent_id, 1]
                sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist
                
                # Check if agent observes poi, update closest step distance
                if sqr_distance < sqr_observation_rad:
                    observer_count += 1
                    if sqr_distance < closest_rover_sqr_dist:
                        closest_rover_sqr_dist = sqr_distance
                        
                        
            # update closest distance only if poi is observed    
            if observer_count >= coupling:
                if closest_rover_sqr_dist < closest_obs_dist:
                    closest_obs_dist = closest_rover_sqr_dist
        
        # add to global reward if poi is observed 
        if closest_obs_dist < sqr_observation_rad:
            if closest_obs_dist < min_sqr_dist:
                closest_obs_dist = min_sqr_dist
            g_reward += poi_values[poi_id] / closest_obs_dist

    
    for agent_id in range(number_agents):
        g_without_self = 0

        for poi_id in range(number_pois):
            closest_obs_dist = inf

            for step_number in range(total_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                closest_rover_sqr_dist = inf

                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        # Calculate separation distance between poi and agent\
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist
                        
                        # Check if agent observes poi, update closest step distance
                        if sqr_distance < sqr_observation_rad:
                            observer_count += 1
                            if sqr_distance < closest_rover_sqr_dist:
                                closest_rover_sqr_dist = sqr_distance
                            
                            
                # update closest distance only if poi is observed    
                if observer_count >= coupling:
                    if closest_rover_sqr_dist < closest_obs_dist:
                        closest_obs_dist = closest_rover_sqr_dist
            
            # add to global reward if poi is observed 
            if closest_obs_dist < sqr_observation_rad:
                if closest_obs_dist < min_sqr_dist:
                    closest_obs_dist = min_sqr_dist
                g_without_self += poi_values[poi_id] / closest_obs_dist
        difference_reward[agent_id] = g_reward - g_without_self
        
    data["Agent Rewards"] = d_reward
    data["Global Reward"] = g_reward
    

# D++ REWARD ----------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_dpp_reward(data):
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs'] 
    cdef double min_sqr_dist = data["Minimum Distance"] ** 2
    cdef int total_steps = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double sqr_observation_rad = data["Observation Radius"] ** 2
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactualCount
    cdef double agent_x_dist, agent_y_dist, closest_obs_dist, sqr_distance, closest_rover_sqr_dist
    cdef double inf = float("inf")
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    
    dpp_reward = np.zeros(number_agents)
    cdef double[:] dplusplus_reward = dpp_reward
    
    # Calculate Global Reward
    for poi_id in range(number_pois):
        closest_obs_dist = inf

        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            closest_rover_sqr_dist = inf

            for agent_id in range(number_agents):
                # Calculate separation distance between poi and agent
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, agent_id, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, agent_id, 1]
                sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist
                
                # Check if agent observes poi, update closest step distance
                if sqr_distance < sqr_observation_rad:
                    observer_count += 1
                    if sqr_distance < closest_rover_sqr_dist:
                        closest_rover_sqr_dist = sqr_distance
                        
                        
            # update closest distance only if poi is observed    
            if observer_count >= coupling:
                if closest_rover_sqr_dist < closest_obs_dist:
                    closest_obs_dist = closest_rover_sqr_dist
        
        # add to global reward if poi is observed 
        if closest_obs_dist < sqr_observation_rad:
            if closest_obs_dist < min_sqr_dist:
                closest_obs_dist = min_sqr_dist
            g_reward += poi_values[poi_id] / closest_obs_dist
            
    # Calculate Difference Reward      
    for agent_id in range(number_agents):
        g_without_self = 0

        for poi_id in range(number_pois):
            closest_obs_dist = inf

            for step_number in range(total_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                closest_rover_sqr_dist = inf

                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        # Calculate separation distance between poi and agent\
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist
                        
                        # Check if agent observes poi, update closest step distance
                        if sqr_distance < sqr_observation_rad:
                            observer_count += 1
                            if sqr_distance < closest_rover_sqr_dist:
                                closest_rover_sqr_dist = sqr_distance


                # update closest distance only if poi is observed    
                if observer_count >= coupling:
                    if closest_rover_sqr_dist < closest_obs_dist:
                        closest_obs_dist = closest_rover_sqr_dist
            
            # add to global reward if poi is observed 
            if closest_obs_dist < sqr_observation_rad:
                if closest_obs_dist < min_sqr_dist:
                    closest_obs_dist = min_sqr_dist
                g_without_self += poi_values[poi_id] / closest_obs_dist
        dplusplus_reward[agent_id] = g_reward - g_without_self
    
    # Calculate Dpp Reward
    for counterfactualCount in range(coupling):

        # Calculate Difference with Extra Me Reward
        for agent_id in range(number_agents):
            g_with_counterfactuals = 0

            for poi_id in range(number_pois):
                closest_obs_dist = inf

                for step_number in range(total_steps):
                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    closest_rover_sqr_dist = inf

                    for other_agent_id in range(number_agents):
                        # Calculate separation distance between poi and agent\
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist
                        
                    
                        if sqr_distance < sqr_observation_rad:
                            # Check if agent observes poi, update closest step distance
                            observer_count += 1 + ((agent_id == other_agent_id) * counterfactualCount)
                            if sqr_distance < closest_rover_sqr_dist:
                                closest_rover_sqr_dist = sqr_distance

                    # update closest distance only if poi is observed    
                    if observer_count >= coupling:
                        if closest_rover_sqr_dist < closest_obs_dist:
                            closest_obs_dist = closest_rover_sqr_dist
                
                # add to global reward if poi is observed 
                if closest_obs_dist < sqr_observation_rad:
                    if closest_obs_dist < min_sqr_dist:
                        closest_obs_dist = min_sqr_dist
                    g_with_counterfactuals += poi_values[poi_id] / closest_obs_dist
            dplusplus_reward[agent_id] = max(dplusplus_reward[agent_id],
            (g_with_counterfactuals - g_reward)/(1.0 + counterfactualCount))
        
    data["Agent Rewards"] = dpp_reward
    data["Global Reward"] = g_reward
    

# D++ WITH SUPERVISOR REWARD ----------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_dpp_sup_reward(data):
    cdef int number_agents = data['Number of Agents']
    cdef int number_pois = data['Number of POIs']
    cdef double min_sqr_dist = data["Minimum Distance"] ** 2
    cdef int total_steps = data["Steps"] + 1
    cdef int coupling = data["Coupling"]
    cdef double sqr_observation_rad = data["Observation Radius"] ** 2
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactual_count
    cdef double agent_x_dist, agent_y_dist, closest_obs_dist, sqr_distance, closest_rover_sqr_dist
    cdef double inf = float("inf")
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added

    dpp_reward = np.zeros(number_agents)
    cdef double[:] dplusplus_reward = dpp_reward

    # CALCULATE GLOBAL REWARD ---------------------------------------------------------------
    for poi_id in range(number_pois):
        closest_obs_dist = inf

        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            closest_rover_sqr_dist = inf

            for agent_id in range(number_agents):
                # Calculate separation distance between poi and agent
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, agent_id, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, agent_id, 1]
                sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist

                # Check if agent observes poi, update closest step distance
                if sqr_distance < sqr_observation_rad:
                    observer_count += 1
                    if sqr_distance < closest_rover_sqr_dist:
                        closest_rover_sqr_dist = sqr_distance


            # update closest distance only if poi is observed
            if observer_count >= coupling:
                if closest_rover_sqr_dist < closest_obs_dist:
                    closest_obs_dist = closest_rover_sqr_dist

        # add to global reward if poi is observed
        if closest_obs_dist < sqr_observation_rad:
            if closest_obs_dist < min_sqr_dist:
                closest_obs_dist = min_sqr_dist
            g_reward += poi_values[poi_id] / closest_obs_dist

    # CALCULATE DIFFERENCE REWARD --------------------------------------------------------------
    for agent_id in range(number_agents):
        g_without_self = 0

        for poi_id in range(number_pois):
            closest_obs_dist = inf

            for step_number in range(total_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                closest_rover_sqr_dist = inf

                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        # Calculate separation distance between poi and agent\
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist

                        # Check if agent observes poi, update closest step distance
                        if sqr_distance < sqr_observation_rad:
                            observer_count += 1
                            if sqr_distance < closest_rover_sqr_dist:
                                closest_rover_sqr_dist = sqr_distance


                # update closest distance only if poi is observed
                if observer_count >= coupling:
                    if closest_rover_sqr_dist < closest_obs_dist:
                        closest_obs_dist = closest_rover_sqr_dist

            # add to global reward if poi is observed
            if closest_obs_dist < sqr_observation_rad:
                if closest_obs_dist < min_sqr_dist:
                    closest_obs_dist = min_sqr_dist
                g_without_self += poi_values[poi_id] / closest_obs_dist
        dplusplus_reward[agent_id] = g_reward - g_without_self

    # CALCULATE DPP REWARD -----------------------------------------------------------------------
    for counterfactual_count in range(coupling):

        # Calculate Difference with Extra Me Reward
        for agent_id in range(number_agents):
            g_with_counterfactuals = 0

            for poi_id in range(number_pois):
                closest_obs_dist = inf

                for step_number in range(total_steps):
                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    closest_rover_sqr_dist = inf

                    for other_agent_id in range(number_agents):
                        # Calculate separation distance between poi and agent\
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        sqr_distance = agent_x_dist * agent_x_dist + agent_y_dist * agent_y_dist


                        if sqr_distance < sqr_observation_rad:
                            # Check if agent observes poi, update closest step distance
                            observer_count += 1 + ((agent_id == other_agent_id) * counterfactual_count)
                            if sqr_distance < closest_rover_sqr_dist:
                                closest_rover_sqr_dist = sqr_distance

                    # update closest distance only if poi is observed
                    if observer_count >= coupling:
                        if closest_rover_sqr_dist < closest_obs_dist:
                            closest_obs_dist = closest_rover_sqr_dist

                # add to global reward if poi is observed
                if closest_obs_dist < sqr_observation_rad:
                    if closest_obs_dist < min_sqr_dist:
                        closest_obs_dist = min_sqr_dist
                    g_with_counterfactuals += poi_values[poi_id] / closest_obs_dist
            dplusplus_reward[agent_id] = max(dplusplus_reward[agent_id],
            (g_with_counterfactuals - g_reward)/(1.0 + counterfactual_count))

    data["Agent Rewards"] = dpp_reward
    data["Global Reward"] = g_reward