from envs.rover_domain.env_wrapper import RoverDomainPython

env = RoverDomainPython(args, NUM_EVALS)

args is a class with 


self.dim_x = WORLD_SIZE; 
self.obs_radius = OBSERVABILITY
self.act_dist = HOW FAR THE ROVER NEEDS TO BE IN ORDER TO ACTIVATE A POI
self.angle_res = ANGLE_RESOLUTION
self.num_poi = NUM_POIS 
self.num_agents = NUM_AGENTS 
self.ep_len = EPISODE_LENGTH
self.poi_rand = INITIALIZE POI RANDOMLY? 
self.coupling = COUPLING 
self.rover_speed = SPEED OF ROVER (default is 1)
self.sensor_model = 'Closest': report the closest distance 
                    'Density': report the average (standard rover domain)




NUM_EVALS = Number of ROverDomain to run in parallel [Similar to a parallel universe - a policy will act in multiple universes (env) concurrently using the same feedforward operation] - massive speedup

