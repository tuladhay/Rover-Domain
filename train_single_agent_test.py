import learning.MLP as MLP
import rover_domain
import multiprocessing
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate_policy(p):
    """
    Creates a new world to evaluate a policy in, and evaluates the policy in said world.
    :param p: The policy to test
    :return: (Integer) The fitness (G score) of the policy
    """
    rd = rover_domain.RoverDomain()
    rd.poi_positions = np.array([10, 10])
    done = False
    state = rd.rover_observations
    reward = 0
    while not done:
        state = np.array(state)
        state = torch.tensor(state.flatten())
        action = p.forward(state.float())
        state, r, done, _ = rd.step(np.array([action.detach().numpy()], dtype='double'))
        reward += np.array(r)[0]
    return reward


def main():
    pool = multiprocessing.Pool()
    # results_df = pd.DataFrame(columns=["Generation", "Best Score"])
    best_results = []
    # Init population
    policies = []
    # Population size of 50
    for _ in range(50):
        policies.append(MLP.Policy(8, 128, 2))

    # 5000 generations
    for gen in range(5000):
        print("Generation: ", gen)
        rewards = pool.map(evaluate_policy, policies)
        # sort them and evolve & mutate them
        results = sorted(zip(list(range(len(policies))), rewards), key=lambda x:x[1], reverse=True)
        results = results[0]
        # copy best results
        for r in results[20:40]:
            copy_state_dict = policies[random.choice(results[:20])].state_dict()
            policies[r].load_state_dict(copy_state_dict)
            policies[r].mutate()
        for r in results[40:]:
            # Inject random policies
            policies[r] = MLP.Policy(8, 128, 2)
        # Record best data so far
        best_results.append(max(rewards))
        # results_df.append({"Generation": gen, "Best Score": max(rewards)}, ignore_index=True)
    # print(results_df)
    # results_df.plot()
    plt.plot(best_results)
    plt.savefig("./Single_agent_test.png")


main()
