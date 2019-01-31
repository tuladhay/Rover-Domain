import rover_domain_w_setup as rover_domain
import numpy as np
import pandas as pd


def test_G():
    """
    Creates a rover domain, initializes it, and uses CCEA to train a team using the global reward signal.
    :return: None (at the moment)
    """
    rd = rover_domain.RoverDomain()

    for t in range(rd.n_steps):
        # For each step
        actions = np.random.random((2, 1))
        print(actions)
        obs, reward, done, _ = rd.step(actions)
        print(obs, reward, done)



if __name__ == "__main__":
    test_G()