import learning.MLP as MLP
import rover_domain
import multiprocessing

def evaluate_policy(p):
    """
    Creates a new world to evaluate a policy in, and evaluates the policy in said world.
    :param p: The policy to test
    :return: (Integer) The fitness (G score) of the policy
    """
    rd = rover_domain.RoverDomain()
    done = False
    state = rd.rover_observations
    reward = 0
    while not done:
        action = p.forward(state)
        state, r, done, _ = rd.step(action)
        reward += r
    return reward





