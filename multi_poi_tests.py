import multi_poi_rover_domain
import numpy as np
import pandas as pd
import learning.MLP as MLP
import multiprocessing
import torch
import random

def evaluate_policy(policies):
    """
    Creates a new world, evaluates set of policies p on the world.
    :param policies: the set of policies (agents) to try the world
    :return: Rewards for each agent
    """
    rd = multi_poi_rover_domain.SequentialPOIRD()
    rd.poi_positions = np.array([[0, 10], [8, 10], [16, 10]], dtype="double")
    done = False
    state = rd.rover_observations
    if not done:
        actions = []
        for i, p in enumerate(policies):
            s = np.array(state[i])
            s = torch.tensor(s.flatten())
            with torch.set_grad_enabled(False):
                actions.append(np.array(p.forward(s.float())))
        actions = np.array(actions, dtype="double")
        state, reward, done, _ = rd.step(actions)
        # Updates the sequence map
        rd.update_sequence_visits()
    return [rd.sequential_score()]*len(policies)


class Agent:
    def __init__(self, pool_size=50):
        self.policy_pool = []
        self.cum_rewards = [0]*pool_size
        for _ in range(pool_size):
            self.policy_pool.append(MLP.Policy(16, 64, 2))

    def reset(self):
        self.cum_rewards = [0]*len(self.cum_rewards)


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    agents = []
    best_performance = []
    for _ in range(3):
        agents.append(Agent())

    with torch.set_grad_enabled(False):
        for gen in range(5000):
            teams = []
            for _ in agents:
                teams.append(list(range(50)))
                random.shuffle(teams[-1])
            teams = np.array(teams)
            teams = teams.transpose()
            policy_teams = []
            for t in teams:
                p = [agents[0].policy_pool[t[0]],
                     agents[1].policy_pool[t[1]],
                     agents[2].policy_pool[t[2]]]
                policy_teams.append(p)

            team_performances = pool.map(evaluate_policy, policy_teams)
            # Update the cumulative performance of each policy
            for i, t in enumerate(teams):
                for a in range(len(agents)):
                    agents[a].cum_rewards[t[a]] += team_performances[i][a]
            print("Gen {} best team: ".format(gen), max(team_performances))
            best_performance.append(max(team_performances)[0])

            # Rank and update each agent population
            if gen % 10 == 0:
                for a in agents:
                    # Gets the keys for policy sorted by highest cumulative score first
                    results = sorted(zip(list(range(len(a.policy_pool))), a.cum_rewards), key=lambda x:x[1], reverse=True)
                    results = results[0]
                    for r in results[25:50]:
                        copy_state_dict = a.policy_pool[random.choice(results[:25])].state_dict()
                        a.policy_pool[r].load_state_dict(copy_state_dict)
                        a.policy_pool[r].mutate()
                    for r in results[45:]:
                        # Inject random policies into a.policy_pool
                        a.policy_pool[r] = MLP.Policy(16, 64, 2)
                    # zero out the score again
                    a.reset()
    best_performance = pd.DataFrame(best_performance)
    best_performance.to_hdf("./G_multi-reward_best.hdf5")





