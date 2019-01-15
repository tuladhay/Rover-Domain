#! /usr/bin/python3.6

from rover_domain_core_gym import rover_domain_core_gym
from parameters import parameters as p
from mods import mod as m
from code.agent_domain import get_agent_actions
from code.trajectory_history import create_trajectory_histories, save_trajectory_histories, update_trajectory_histories
from code.reward_history import save_reward_history, create_reward_history, update_reward_history
from code.ccea import *  # CCEA

step_count = 5
generations_per_episode = 3
tests_per_episode = 1
num_episodes = 20

# NOTE: Add the mod functions (variables) to run to mod_col here:
mod_col = [
    m.global_reward_mod,
    m.difference_reward_mod,
    m.dpp_reward_mod
]

i = 0
while i < 1:
    print("Run %i" % (i))

    for func in mod_col:
        sim = rover_domain_core_gym()
        func(p.data)

        # Trial Begins
        create_reward_history(p.data)
        init_ccea(num_inputs=8, num_outputs=2, num_units=32)(p.data)
        p.data["Steps"] = step_count

        for episodeIndex in range(num_episodes):
            p.data["Episode Index"] = episodeIndex

            # Training Phase
            obs = sim.reset('Train', True)

            for world_index in range(generations_per_episode):
                p.data["World Index"] = world_index
                obs = sim.reset('Train', False)
                assign_ccea_policies(p.data)

                done = False
                step_count = 0
                while not done:
                    get_agent_actions(p.data)
                    joint_action = p.data["Agent Actions"]
                    obs, reward, done, info = sim.step(joint_action)
                    step_count += 1

                reward_ccea_policies(p.data)

            # Testing Phase
            obs = sim.reset('Test', True)
            assign_best_ccea_policies(p.data)

            for world_index in range(tests_per_episode):
                p.data["World Index"] = world_index
                obs = sim.reset('Test', False)

                done = False
                step_count = 0
                while not done:
                    get_agent_actions(p.data)
                    joint_action = p.data["Agent Actions"]
                    obs, reward, done, info = sim.step(joint_action)
                    step_count += 1

            evolve_ccea_policies(p.data)
            update_reward_history(p.data)

            # Trial End
            save_reward_history(p.data)
            save_trajectory_histories(p.data)

    i += 1
