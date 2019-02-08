import rover_domain_w_setup as r

from ccea import CCEA
import numpy as np
from tensorboardX import SummaryWriter
import time
from pathlib import Path
import os
import csv


class Params:
    def __init__(self):
        self.population_size = 10
        self.n_agents = None
        self.mutation_rate = 0.01

        # For Neural Network Policies
        self.nn_input_size = None
        self.nn_output_size = None
        self.nn_hidden_size = 16

def get_env_setting():
    setting = {"communication" : env.comm_acs,
               "n_agents" : env.n_rovers,
               "n_pois": env.n_pois,
               "n_req": env.n_req,
               "n_steps": env.n_steps,
               "setup_size": env.setup_size,
               "min_dist": env.min_dist,
               "interaction_dist": env.interaction_dist,
               "reorients": env.reorients,
               "discounts_eval": env.discounts_eval,
               "n_obs_sections": env.n_obs_sections,
                "timestr": timestr
                }
    return setting


if __name__=="__main__":

    # Initialize the environment
    env = r.RoverDomain()
    env.reset()

    # Initialize params for CCEA
    params = Params()
    params.n_agents = env.n_rovers
    params.nn_input_size = env.rover_observations.base[0].size
    params.nn_output_size = 2  # todo: Set this automatically

    # Communication
    use_comm = True
    if use_comm:
        params.nn_output_size += env.n_obs_sections  # same num of channels as obs quadrants

    # -----------------------------------------------------------------------------------------------------------------#
    # Logger and save experiment setting
    # -----------------------------------------------------------------------------------------------------------------#
    timestr = time.strftime("__%m%d-%H%M%S")
    file_path = Path('./Experiments') / ("R"+str(env.n_rovers) + "-P"+str(env.n_pois) + "-Cp"+str(env.n_req) + timestr)
    os.makedirs(file_path)

    # Save setting
    setting = get_env_setting()
    with open(str(file_path) +'/settings.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in setting.items():
            writer.writerow([key, value])

    logger = SummaryWriter(str(file_path))

    # -----------------------------------------------------------------------------------------------------------------#
    # Run Algorithm
    # -----------------------------------------------------------------------------------------------------------------#

    # Initialize algorithm
    ccea = CCEA(params)
    test_team_runs = 5      # eval runs for best_team

    episodes = 10000
    for ep_i in range(episodes):
        # Makes population of 2*K policies for M agents
        ccea.mutate()
        # reset team builder. Makes team without replacement for leniency.
        ccea.reset_teambuilder()

        # This evaluation loop is for CCEA (see lecture slides). Leniency evals * mutated population size
        for _ in range(ccea.leniency_evals*len(ccea.agents[0].population)):  # run n number of times for leniency evaluation.
            # List is popped, so will run out if make_team is run more than ccea.leniency eval times.
            env.poi_positions = None        # Hack
            env.reset()
            ccea.make_team()

            done = False
            # --- Run entire trajectory using this team policy --- #
            while not done:
                # List of observations for list of agents
                joint_obs = [env.rover_observations.base[i].flatten() for i in range(params.n_agents)]

                # List of actions for a list of agents. Actions stored in ccea.joint_action
                ccea.get_team_action(joint_obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])

                # Step
                _, _, done, _ = env.step(agent_actions)  # obs, step_rewards, done, self

            # Reward after running entire trajectory
            env.update_rewards_traj_global_eval()
            fitness = env.rover_rewards[0]      # All agents have same global reward.
            ccea.assign_fitness(fitness)        # Uses max(...) for Leniency

        # selection. Back to K policies for M agents. Also makes a best_policy team
        ccea.selection()

        # ------------------------------------------------------------------------------------------------------------#
        # Evaluate the best team. This is the fitness that we record for learning
        # ------------------------------------------------------------------------------------------------------------#
        ccea.team = ccea.best_team
        fitness = []        # assuming fitness is always positive
        for _ in range(test_team_runs):
            env.poi_positions = None        # Hack
            env.reset()
            done = False
            while not done:
                joint_obs = [env.rover_observations.base[i].flatten() for i in range(params.n_agents)]
                ccea.get_team_action(joint_obs)
                agent_actions = np.array([np.double(ac.data[0]) for ac in ccea.joint_action])
                _, _, done, _ = env.step(agent_actions)  # obs, step_rewards, done, self
            env.update_rewards_traj_global_eval()
            fitness.append(env.rover_rewards[0])  # All agents have same global reward.
        fitness = np.mean(fitness)      # averaging over eval runs
        # Logging
        logger.add_scalar('mean_team_fitness', fitness, ep_i)

        # Save model
        if not ep_i % 100:
            os.makedirs(file_path / 'models', exist_ok=True)
            ccea.save(file_path / 'models' / ('model_ep%i.pt' % (ep_i)))

        # Print Episode and fitness
        print("Episode:"+str(ep_i) + "  Fitness:" + str(fitness))

    print()
