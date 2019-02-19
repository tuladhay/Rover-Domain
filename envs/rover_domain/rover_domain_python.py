import random, sys
from random import randint
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import softmax


class RoverDomain:

    def __init__(self, args):

        self.args = args
        self.dim_y = self.args.dim_x

        # Gym compatible attributes
        self.observation_space = np.zeros((1, int(2*360 / self.args.angle_res)))
        self.action_space = np.zeros((1, self.args.action_dim))

        self.istep = 0  # Current Step counter

        # Initialize POI containers tha track POI position and status
        self.poi_pos = [[None, None] for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][x, y] coordinate
        self.poi_status = [False for _ in range(self.args.num_poi)]  # FORMAT: [poi_id][status] --> [T/F] is observed?
        self.poi_value = [float(i+1) for i in range(self.args.num_poi)]  # FORMAT: [poi_id][value]?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.args.num_agents)]  # FORMAT: [rover_id][x, y] coordinate

        # Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos] # FORMAT: [rover_id][timestep][x, y]
        self.action_seq = [[0.0 for _ in range(self.args.action_dim)] for _ in range(self.args.num_agents)] # FORMAT: [timestep][rover_id][action]

        # End of an episode
        self.done = False

        # Communication
        self.comm_dim = args.n_comm_bits
        # Joint state w comm
        #self.comm_state = [[0.0 for _ in range(12)] for _ in range(args.num_agents)]  # TODO: do this automatically
        self.comm_state = np.zeros(self.comm_dim)  #[0.0 for _ in range(args.n_comm_bits)]  # TODO: do this automatically

    def reset(self):
        self.done = False
        self.reset_poi_pos()
        self.reset_rover_pos()
        self.poi_value = [float(i+1) for i in range(self.args.num_poi)]
        self.poi_status = [False for _ in range(self.args.num_poi)]
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.args.action_dim)] for _ in range(self.args.num_agents)]
        self.istep = 0
        # Reset communication bits to zeros
        self.comm_state = np.zeros(self.comm_dim)
        return self.get_joint_state()  # [rover_densities, poi_densities]

    def view_pos(self):
        ''' Function to plot the current positions of POIs and Rovers '''
        poi_x = [pos[0] for pos in self.poi_pos]
        poi_y = [pos[1] for pos in self.poi_pos]
        rov_x = [pos[0] for pos in self.rover_pos]
        rov_y = [pos[1] for pos in self.rover_pos]
        plt.title("blue:poi, red:rover")
        plt.scatter(poi_x, poi_y, color='blue')
        plt.scatter(rov_x, rov_y, color='red')
        plt.show()

    def step(self, joint_action):
        self.istep += 1
        comm_signal = np.zeros(self.comm_dim)

        for rover_id in range(self.args.num_agents):
            action = joint_action[rover_id]
            self.rover_pos[rover_id] = [self.rover_pos[rover_id][0]+action[0], self.rover_pos[rover_id][1]+action[1]]  # Execute action

            # COMMUNICATION = add the softmax for each agent's comm channel
            comm_signal += softmax(action[2:], axis=None)  # actions after the 2 physical actions are communication actions

        #Append rover path
        for rover_id in range(self.args.num_agents):
            self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]))

        #Compute done
        self.done = self.istep >= self.args.ep_len

        #info
        global_reward = None
        if self.done:
            global_reward = self.get_global_traj_reward()

        # Do this so that get_joint_state() can append it to other states
        self.comm_state = comm_signal

        return self.get_joint_state(), self.get_local_reward(), self.done, global_reward

    def reset_poi_pos(self):
        start = 0.0
        end = self.args.dim_x - 1.0
        rad = int(self.args.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        if self.args.poi_rand: #Random
            for i in range(self.args.num_poi):
                if i % 3 == 0:
                    x = randint(start, center - rad - 1)
                    y = randint(start, end)
                elif i % 3 == 1:
                    x = randint(center + rad + 1, end)
                    y = randint(start, end)
                elif i % 3 == 2:
                    x = randint(center - rad, center + rad)
                    if random.random()<0.5:
                        y = randint(start, center - rad - 1)
                    else:
                        y = randint(center + rad + 1, end)

                self.poi_pos[i] = [x, y]

        else: #Not_random
            for i in range(self.args.num_poi):
                if i % 4 == 0:
                    x = start + int(i/4) #randint(start, center - rad - 1)
                    y = start + int(i/3)
                elif i % 4 == 1:
                    x = end - int(i/4) #randint(center + rad + 1, end)
                    y = start + int(i/4)#randint(start, end)
                elif i % 4 == 2:
                    x = start+ int(i/4) #randint(center - rad, center + rad)
                    y = end - int(i/4) #randint(start, center - rad - 1)
                else:
                    x = end - int(i/4) #randint(center - rad, center + rad)
                    y = end - int(i/4) #randint(center + rad + 1, end)
                self.poi_pos[i] = [x, y]


    def reset_rover_pos(self):
        start = 1.0; end = self.args.dim_x - 1.0
        rad = int(self.args.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        # if self.args.unit_test == 1: #Unit test
        #     self.rover_pos[0] = [end,0];
        #     return

        for rover_id in range(self.args.num_agents):
                quadrant = rover_id % 4
                if quadrant == 0:
                    x = center - 1 - (rover_id / 4) % (center - rad)
                    y = center - (rover_id / (4 * center - rad)) % (center - rad)
                if quadrant == 1:
                    x = center + (rover_id / (4 * center - rad)) % (center - rad)-1
                    y = center - 1 + (rover_id / 4) % (center - rad)
                if quadrant == 2:
                    x = center + 1 + (rover_id / 4) % (center - rad)
                    y = center + (rover_id / (4 * center - rad)) % (center - rad)
                if quadrant == 3:
                    x = center - (rover_id / (4 * center - rad)) % (center - rad)
                    y = center + 1 - (rover_id / 4) % (center - rad)
                self.rover_pos[rover_id] = [x, y]

    def get_joint_state(self):
        joint_state = []
        for rover_id in range(self.args.num_agents):
            self_x = self.rover_pos[rover_id][0]
            self_y = self.rover_pos[rover_id][1]

            rover_state = [0.0 for _ in range(int(360 / self.args.angle_res))]
            poi_state = [0.0 for _ in range(int(360 / self.args.angle_res))]
            temp_poi_dist_list = [[] for _ in range(int(360 / self.args.angle_res))]
            temp_rover_dist_list = [[] for _ in range(int(360 / self.args.angle_res))]

            # Log all distance into brackets for POIs
            for loc, status, value in zip(self.poi_pos, self.poi_status, self.poi_value):
                if status == True: continue #If accessed ignore

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist > self.args.obs_radius: continue #Observability radius

                if math.isnan(angle):
                    print(angle)
                    print("*********************************************")
                bracket = int(angle / self.args.angle_res)
                if dist == 0: dist = 0.001
                temp_poi_dist_list[bracket].append((value/(dist*dist)))

            # Log all distance into brackets for other drones
            for id, loc, in enumerate(self.rover_pos):
                if id == rover_id: continue #Ignore self

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist > self.args.obs_radius: continue #Observability radius

                if dist == 0: dist = 0.001
                bracket = int(angle / self.args.angle_res)
                temp_rover_dist_list[bracket].append((1/(dist*dist))) #if rovers overlap in pos,this number becomes 1million


            ####Encode the information onto the state
            for bracket in range(int(360 / self.args.angle_res)):
                # POIs
                num_poi = len(temp_poi_dist_list[bracket])
                if num_poi > 0:
                    if self.args.sensor_model == 'density': poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi #Density Sensor
                    elif self.args.sensor_model == 'closest': poi_state[bracket] = max(temp_poi_dist_list[bracket])  #Closest Sensor
                    else: sys.exit('Incorrect sensor model')
                else: poi_state[bracket] = -1.0

                #Rovers
                num_agents = len(temp_rover_dist_list[bracket])
                if num_agents > 0:
                    if self.args.sensor_model == 'density': rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents #Density Sensor
                    elif self.args.sensor_model == 'closest': rover_state[bracket] = max(temp_rover_dist_list[bracket]) #Closest Sensor
                    else: sys.exit('Incorrect sensor model')
                else: rover_state[bracket] = -1.0

            state = rover_state + poi_state #Append rover and poi to form the full state

            # #Append wall info
            # state = state + [-1.0, -1.0, -1.0, -1.0]
            # if self_x <= self.args.obs_radius: state[-4] = self_x
            # if self.args.dim_x - self_x <= self.args.obs_radius: state[-3] = self.args.dim_x - self_x
            # if self_y <= self.args.obs_radius :state[-2] = self_y
            # if self.args.dim_y - self_y <= self.args.obs_radius: state[-1] = self.args.dim_y - self_y

            #state = np.array(state)
            joint_state.append(state)

        for i, state in enumerate(joint_state):
            joint_state[i] = joint_state[i] + self.comm_state.tolist()

        return joint_state


    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        v1 = x2 - x1
        v2 = y2 - y1
        angle = np.rad2deg(np.arctan2(v1, v2))
        if angle < 0: angle += 360

        dist = v1 * v1 + v2 * v2
        dist = math.sqrt(dist)

        return angle, dist


    def get_local_reward(self):
        #Update POI's visibility
        poi_visitors = [[] for _ in range(self.args.num_poi)]
        for i, loc in enumerate(self.poi_pos): #For all POIs
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            for rover_id in range(self.args.num_agents): #For each rover
                x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
                dist = math.sqrt(x1 * x1 + y1 * y1)
                if dist <= self.args.act_dist: poi_visitors[i].append(rover_id) #Add rover to POI's visitor list

        #Compute reward
        rewards = [0.0 for _ in range(self.args.num_agents)]
        for poi_id, rovers in enumerate(poi_visitors):
            if len(rovers) >= self.args.coupling:
                self.poi_status[poi_id] = True
                lucky_rovers = random.sample(rovers, self.args.coupling)
                for rover_id in lucky_rovers:
                    rewards[rover_id] += self.poi_value[poi_id]

        return rewards

    def get_global_traj_reward(self):
        if not self.done:
            sys.exit('Global trajectory reward queried before end of episode, check:self.done')

        global_traj_reward = 0
        poi_evals = np.zeros(self.args.num_poi)

        # loop over time-steps
        for tstep in range(self.args.ep_len):
            # rover positions at this time-step
            tstep_rover_pos = [self.rover_path[rover][tstep] for rover in range(self.args.num_agents)]

            for poi_id in range(self.args.num_poi):
                poi_evals[poi_id] = max(poi_evals[poi_id], self.calc_step_eval_from_poi(poi_id, tstep_rover_pos))

        for poi_id in range(self.args.num_poi):
            global_traj_reward += poi_evals[poi_id]

        return global_traj_reward

    def calc_step_eval_from_poi(self, poi_id, tstep_rover_pos):
        sqr_dists_to_poi = np.zeros(self.args.num_agents)

        for rover_id in range(self.args.num_agents):
            displ_x = (tstep_rover_pos[rover_id][0] - self.poi_pos[poi_id][0])
            displ_y = (tstep_rover_pos[rover_id][1] - self.poi_pos[poi_id][1])
            sqr_dists_to_poi[rover_id] = displ_x*displ_x + displ_y*displ_y

        # Sort (n_req) closest rovers for evaluation.
        # Sqr_dists_to_poi is no longer in rover order!
        sqr_dists_to_poi = np.sort(sqr_dists_to_poi)

        # Is there (n_req) rovers observing? Only need to check the (n_req)th closest rover.
        if (sqr_dists_to_poi[self.args.coupling-1] >
                self.args.act_dist * self.args.act_dist):
            # Not close enough?, then there is no reward for this POI
            return 0.

        # Close enough! Continue evaluation.

        # if self.discounts_eval:
        #     sqr_dist_sum = 0.
        #     # Get sum sqr distance of nearest rovers.
        #     for near_rover_id in range(self.n_req):
        #         sqr_dist_sum += sqr_dists_to_poi[near_rover_id]
        #     return self.poi_values[poi_id] / max(self.min_dist, sqr_dist_sum)
        # # Do not discount POI evaluation
        # else:
        #     return self.poi_values[poi_id]

        return self.poi_value[poi_id]


    def render(self):
        # Visualize
        grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]

        # Draw in rover path
        for rover_id, path in enumerate(self.rover_path):
            for loc in path:
                x = int(loc[0]); y = int(loc[1])
                if x < self.args.dim_x and y < self.args.dim_y and x >=0 and y >=0:
                    grid[x][y] = str(rover_id)

        # Draw in food
        for poi_pos, poi_status in zip(self.poi_pos, self.poi_status):
            x = int(poi_pos[0])
            y = int(poi_pos[1])
            marker = '$' if poi_status else '#'
            grid[x][y] = marker

        for row in grid:
            print(row)
        print()

        print('------------------------------------------------------------------------')
