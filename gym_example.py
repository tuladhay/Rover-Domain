#! /usr/bin/python3.6

from rover_domain_core_gym import RoverDomainCore
from parameters import Parameters as p
from mods import Mod as m
from code.agent import get_agent_state, get_agent_actions
from code.trajectory_history import save_trajectory_histories
from code.reward_history import save_reward_history, create_reward_history, update_reward_history
from code.ccea import Ccea
from code.neural_network import NeuralNetwork

# NOTE: Add the mod functions (variables) to run to mod_col here:
mod_col = [
    m.global_reward_mod,
    m.difference_reward_mod,
    m.dpp_reward_mod
]

sim = RoverDomainCore()
cc = Ccea()
nn = NeuralNetwork()

for srun in range(p.stat_runs):
    print("Run: %i" % srun)

    for func in mod_col:
        func(sim.data)
        create_reward_history(sim.data)

        # Trial Begins
        cc.reset_populations()
        nn.reset_nn()

        # Training Phase
        sim.reset('Train', True) # Fully resets rover domain (agent and POI positions/values)

        for gen in range(p.generations):
            #print("Current Gen: %i" % gen)
            sim.reset('Train', False)
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

            # Testing Phase (STILL WORKING ON THIS)
            # sim.reset('Test', True)  # Set mode to test and re-initialize world

            for test in range(p.tests_per_gen):
                sim.data["World Index"] = test
                sim.reset('Test', False)  # Set mode to test and do not reset the world
                get_agent_state(sim.data)
                joint_state = sim.data["Agent Observations"]

                done = False
                step_count = 0
                while not done:
                    for rover_id in range(p.number_of_agents):
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                    get_agent_actions(sim.data, nn.out_layer)
                    obs, reward, done, info = sim.step()
                    step_count += 1

                update_reward_history(sim.data)

    #  Trial End (STILL WORKING ON THIS)
    save_reward_history(sim.data)
    save_trajectory_histories(sim.data)
