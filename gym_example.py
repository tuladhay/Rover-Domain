from parameters import Parameters as p
from rover_domain_core_gym import RoverDomainCore; sim = RoverDomainCore()
from mods import Mod as m
from code.agent import get_agent_actions
from code.trajectory_history import save_trajectory_histories
from code.reward_history import save_reward_history, create_reward_history, update_reward_history
from code.ccea import Ccea; cc = Ccea()
from code.neural_network import NeuralNetwork; nn = NeuralNetwork()

reward_functions = [  # List reward functions you would like to train on (functions execute in order listed)
    m.global_reward_mod,
    m.difference_reward_mod
    #m.dpp_reward_mod
]

for func in reward_functions:
    func(sim.data)
    create_reward_history(sim.data)  # Track performance of best policies for each gen in each stat run

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Initialize environment, ccea, and nn for run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        sim.reset_world('Train', True)  # Fully resets rover domain (agent and POI positions/values)

        for gen in range(p.generations):
            sim.reset_world('Train', False)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            joint_state = sim.data["Agent Observations"]  # State vector

            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                done = False
                step_count = 0
                while done == False:
                    sim.data["Step Index"] = step_count
                    for rover_id in range(p.number_of_agents):
                        policy_id = cc.team_selection[rover_id, team_number]
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    get_agent_actions(sim.data, nn.out_layer)  # Gets outputs from all rover NNs
                    done = sim.step()
                    step_count += 1
                rewards = sim.data["Agent Rewards"]

                # Update fitness of policies using reward information
                for pop_id in range(p.number_of_agents):
                    policy_id = cc.team_selection[pop_id, team_number]
                    cc.fitness[pop_id, policy_id] = rewards[pop_id]

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            # Testing Phase
            # sim.reset_domain('Test', True)  # Set mode to test and fully reset world

            for test in range(p.tests_per_gen):
                sim.reset_world('Test', False)  # Set mode to test and do not fully reset the world
                joint_state = sim.data["Agent Observations"]

                done = False
                step_count = 0
                while done == False:
                    sim.data["Step Index"] = step_count
                    for rover_id in range(p.number_of_agents):
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                    get_agent_actions(sim.data, nn.out_layer)
                    done = sim.step()
                    step_count += 1
                update_reward_history(sim.data, srun)

        save_trajectory_histories(sim.data)

    #  Trial End save data to file
    save_reward_history(sim.data)
    print('\n')
