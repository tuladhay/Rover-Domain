import numpy as np
cimport cython
from parameters import Parameters as p
import math

cdef extern from "math.h":
    double sqrt(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_state_vec(data):  # Calculates join_state_vector for NN input

    cdef int number_agents = p.number_of_agents
    cdef int number_pois = p.number_of_pois
    cdef double min_dist = p.min_distance
    cdef double[:, :] agent_positions = data["Agent Positions"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef double[:, :] agent_orientations = data["Agent Orientations"]
    cdef double[:, :] agent_state = np.zeros((number_agents, 8), dtype = np.float64)
    cdef int agent_id, other_agent_id, poi_id, quadrant
    cdef double x_distance, y_distance
    cdef double dist, angle

    for agent_id in range(number_agents):

        # calculate observation values due to other agents
        for other_agent_id in range(number_agents):

            # agents do not sense self (ergo skip self comparison)
            if agent_id == other_agent_id:
                continue

            # Calculate position vectors between agents with respect to global coordinate frame
            x_distance = agent_positions[other_agent_id, 0] - agent_positions[agent_id, 0] # X-direction
            y_distance = agent_positions[other_agent_id, 1] - agent_positions[agent_id, 1] # Y-direction

            # Calculate distance between POI and Rover and get angle for quadrant
            dist = math.sqrt((x_distance*x_distance)+(y_distance*y_distance))
            angle = math.atan2(y_distance, x_distance)*(180/math.pi)
            if angle < 0:
                angle += 360
            quadrant = int(angle/90)

            # By bounding distance value we implicitly bound sensor values
            if dist < min_dist:
                dist = min_dist

            # other is east of agent
            if quadrant == 0:
                agent_state[agent_id, 0] += 1.0/dist
            elif quadrant == 1: # other is south-east of agent
                agent_state[agent_id, 1] += 1.0/dist
            elif quadrant == 2:  # other is west of agent
                agent_state[agent_id, 2] += 1.0/dist
            else:
                agent_state[agent_id, 3] += 1.0/dist


        # calculate observation values due to pois
        for poi_id in range(number_pois):

            # Calculate position vectors between agent and POI with respect to global coordinate frame
            x_distance = poi_positions[poi_id, 0] - agent_positions[agent_id, 0]
            y_distance = poi_positions[poi_id, 1] - agent_positions[agent_id, 1]

            # Calculate distance between POI and Rover and get angle for quadrant
            dist = math.sqrt((x_distance*x_distance)+(y_distance*y_distance))
            angle = math.atan2(y_distance, x_distance)*(180/math.pi)
            if angle < 0:
                angle += 360
            quadrant = int(angle/90)

            # By bounding distance value we implicitly bound sensor values
            if dist < min_dist:
                dist = min_dist

            if quadrant == 0:
                agent_state[agent_id, 4] += poi_values[poi_id]/dist
            elif quadrant == 1:
                agent_state[agent_id, 5] += poi_values[poi_id]/dist
            elif quadrant == 2:
                agent_state[agent_id, 6] += poi_values[poi_id]/dist
            else:
                agent_state[agent_id, 7] += poi_values[poi_id]/dist

    data["Agent Observations"] = agent_state
    #return agent_state


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_agent_actions(data, nn_output):
    cdef int number_agents = p.number_of_agents
    actions = np.zeros((number_agents, 2), dtype = np.float_)

    cdef int agent_id
    for agent_id in range(number_agents):
        actions[agent_id] = nn_output[agent_id]
    data["Agent Actions"] = actions


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef do_agent_move(data):  # Each agent moves 1 step
    cdef float world_width = p.world_width
    cdef float world_length = p.world_length
    cdef int number_agents = p.number_of_agents
    cdef double[:, :] agent_positions = data["Agent Positions"]
    cdef double[:, :] agent_orientations = data["Agent Orientations"]
    cdef double[:, :] actions = data["Agent Actions"]
    cdef int agent_id
    cdef double dx, dy, norm # Change in x-position, change in y-position, total distance moved

    # move all agents
    for agent_id in range(number_agents):
        # turn action into global frame motion
        dx = actions[agent_id, 0]
        dy = actions[agent_id, 1]
        # print('AGENT: ', dx, dy)

        # globally move and reorient agent
        agent_positions[agent_id, 0] += dx
        agent_positions[agent_id, 1] += dy

        if dx == 0.0 and dy == 0.0:
            agent_orientations[agent_id, 0] = 1.0
            agent_orientations[agent_id, 1] = 0.0
        else:
            norm = sqrt(dx**2 +  dy**2)
            agent_orientations[agent_id, 0] = dx/norm
            agent_orientations[agent_id, 1] = dy/norm

    data["Agent Positions"]  = agent_positions
    data["Agent Orientations"] = agent_orientations