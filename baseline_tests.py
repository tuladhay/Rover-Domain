import numpy as np
import random
import pandas as pd
import rover_domain
import multi_poi_rover_domain
import learning.ccea as ccea
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


def train_G():
    """
    Train on G
    :return: None
    """
    srd = multi_poi_rover_domain.SequentialPOIRD()
    process = ccea.Ccea(srd.n_rovers)



if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)


