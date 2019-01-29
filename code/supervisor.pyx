import math
cimport cython
from parameters import Parameters as p
import numpy as np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def generate_partners(data, tstep, c_number, current_rov_id):
    cdef double[:] rov_distances = np.zeros(p.number_of_agents)
    cdef double[:] partners = np.zeros(c_number)
    cdef double[:, :, :] rover_positions = data["Agent Position History"]
    cdef int rov_id, other_id, i, j
    cdef double agent_x, agent_y, other_agent_x, other_agent_y, x_dist, y_dist, dist

    agent_x = rover_positions[current_rov_id, tstep, 0]
    agent_y = rover_positions[current_rov_id, tstep, 1]
    for other_id in range(p.number_of_agents):
        if current_rov_id != other_id:
            other_agent_x = rover_positions[other_id, tstep, 0]
            other_agent_y = rover_positions[other_id, tstep, 1]

            x_dist = agent_x - other_agent_x
            y_dist = agent_y - other_agent_y
            dist = math.sqrt((x_dist**2)+(y_dist**2))
            rov_distances[other_id] = dist
        if current_rov_id == other_id:
            rov_distances[other_id] = 1000.00

    for i in range(len(rov_distances)):
        j = i + 1
        while j < len(rov_distances):
            if rov_distances[i] > rov_distances[j]:
                rov_distances[i], rov_distances[j] = rov_distances[j], rov_distances[i]
            j += 1

    for rov_id in range(c_number):
        partners[rov_id] = rov_distances[rov_id]

    return partners