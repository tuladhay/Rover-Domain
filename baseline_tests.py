import numpy as np
import random
import pandas as pd
import rover_domain

def test_G():
    """
    Creates a rover domain, initializes it, and uses CCEA to train a team using the global reward signal.
    :return: None (at the moment)
    """
    rd = rover_domain.RoverDomain()
    actions = np.random.random((10, 2))
    print(actions)
    obs, reward, done, _ = rd.step(actions)
    print("observations:", np.array(obs)[:,0])

    for t in range(rd.n_steps):
        # For each step
        actions = np.random.random((10, 2))
        print(actions)
        obs, reward, done, _ = rd.step(actions)
        print(np.array(obs), reward, done)



if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    test_G()