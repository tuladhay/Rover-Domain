import math
cimport cython
from parameters import Parameters as p
import numpy as np
import random

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def closest_others(data, tstep, c_number, current_rov, current_poi):
    cdef double act_distance = p.activation_dist
    cdef double[:] rov_distances = np.zeros(p.number_of_agents)
    cdef int[:] rov_ids = np.zeros(p.number_of_agents, dtype = np.int32)
    cdef double[:] partners = np.zeros(c_number)
    cdef double[:, :, :] rover_positions = data["Agent Position History"]
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int agent_id, other_id, i, j
    cdef double agent_x, agent_y, other_agent_x, other_agent_y, x_dist, y_dist, dist

    agent_x = rover_positions[current_rov, tstep, 0]
    agent_y = rover_positions[current_rov, tstep, 1]
    for other_id in range(p.number_of_agents):
        rov_ids[other_id] = other_id
        if current_rov != other_id:
            other_agent_x = rover_positions[other_id, tstep, 0]
            other_agent_y = rover_positions[other_id, tstep, 1]

            x_dist = agent_x - other_agent_x
            y_dist = agent_y - other_agent_y
            dist = math.sqrt((x_dist**2)+(y_dist**2))
            rov_distances[other_id] = dist
        if current_rov == other_id:
            rov_distances[other_id] = 1000.00

    for i in range(len(rov_distances)):  # Finds closest other rovers to current rover
        j = i + 1
        while j < len(rov_distances):
            if rov_distances[i] > rov_distances[j]:
                rov_distances[i], rov_distances[j] = rov_distances[j], rov_distances[i]
                rov_ids[i], rov_ids[j] = rov_ids[j], rov_ids[i]
            j += 1

    for i in range(c_number):  # Computes distances from closest others to POI
        agent_id = rov_ids[i]
        dist_x = poi_positions[current_poi, 0] - rover_positions[agent_id, tstep, 0]
        dist_y = poi_positions[current_poi, 1] - rover_positions[agent_id, tstep, 1]
        dist = math.sqrt((dist_x**2) + (dist_y**2))
        partners[i] = dist

    return partners

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def random_partners(data, tstep, c_number, current_rov, current_poi):
    cdef double act_distance = p.activation_dist
    cdef double[:] partners = np.zeros(c_number)
    cdef double[:, :, :] rover_positions = data["Agent Position History"]
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int partner_id, i
    cdef double agent_x, agent_y, x_dist, y_dist, dist


    for i in range(c_number):  # Computes distances from closest others to POI
        partner_id = random.randint(0, p.number_of_agents-1)
        while partner_id == current_rov:
            partner_id = random.randint(0, p.number_of_agents-1)

        agent_x = rover_positions[partner_id, tstep, 0]
        agent_y = rover_positions[partner_id, tstep, 1]
        x_dist = poi_positions[current_poi, 0] - agent_x
        y_dist = poi_positions[current_poi, 1] - agent_y
        dist = math.sqrt((x_dist**2)+(y_dist**2))
        partners[i] = dist

    return partners

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def value_based_partners(data, c_number, current_poi):
    cdef double[:] partners = np.zeros(c_number)
    cdef double[:] poi_values = data["Poi Values"]
    cdef int i
    cdef double partner_value
    cdef double max_poi_val = 10.0


    for i in range(c_number):  # Computes distances from closest others to POI
        partner_value = max_poi_val/poi_values[current_poi]
        partners[i] = partner_value

    return partners
