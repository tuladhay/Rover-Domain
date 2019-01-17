#! /usr/bin/python3.6

from rover_domain_core_gym import RoverDomainCore
from parameters import Parameters as p
from mods import Mod as m
from code.agent import get_agent_state, get_agent_actions
from code.trajectory_history import create_trajectory_histories, save_trajectory_histories, update_trajectory_histories
from code.reward_history import save_reward_history, create_reward_history, update_reward_history
from code.ccea import Ccea
from code.neural_network import NeuralNetwork

# NOTE: Add the mod functions (variables) to run to mod_col here:
mod_col = [
    m.global_reward_mod
    #m.difference_reward_mod,
    #m.dpp_reward_mod
]

sim = RoverDomainCore()
cc = Ccea()
nn = NeuralNetwork()

for func in mod_col:
    func(sim.data)

    for s in range(p.stat_runs):
        print("Run: %i" % s)

        # Trial Begins
        create_reward_history(sim.data)
        sim.data["Steps"] = p.total_steps
        cc.reset_populations()
        nn.reset_nn()

        # Training Phase
        obs = sim.reset('Train', True) # Fully resets rover domain (agent and POI positions/values)

        for gen in range(p.generations):
            print("Current Gen: %i" % gen)
            obs = sim.reset('Train', False)
            cc.create_new_pop()  # Create a new population via mutation
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            get_agent_state(sim.data)  # Create state vector for NN inputs (might be redundant here)
            joint_state = sim.data["Agent Observations"]  # State vector

            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(p.number_of_agents):
                    policy_id = cc.team_selection[rover_id, team_number]
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                get_agent_actions(sim.data, nn.out_layer)  # Gets outputs from all rover NNs

                done = False
                step_count = 0
                reward = []
                while not done:
                    obs, reward, done, info = sim.step()
                    step_count += 1

                # Update fitness of policies using reward information
                for pop_id in range(p.number_of_agents):
                    policy_id = cc.team_selection[pop_id, team_number]
                    cc.fitness[pop_id, policy_id] = reward[pop_id]

            cc.down_select()  # Perform down_selection after each policy has been evaluated



        # # Testing Phase (STILL WORKING ON THIS)
        # obs = sim.reset('Test', True)
        #
        # for test in range(tests_per_episode):
        #     sim.data["World Index"] = test
        #     obs = sim.reset('Test', False)
        #
        #     done = False
        #     step_count = 0
        #     while not done:
        #         get_agent_actions(sim.data)
        #         joint_action = sim.data["Agent Actions"]
        #         obs, reward, done, info = sim.step(joint_action)
        #         step_count += 1

        # update_reward_history(sim.data)

        # Trial End (STILL WORKING ON THIS)
        # save_reward_history(sim.data)
        # save_trajectory_histories(sim.data)
