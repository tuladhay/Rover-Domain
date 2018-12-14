"""
An example using the rover domain gym-style interface and the standard, included CCEA learning algorithms.
This is a minimal example, showing the minimal Gym interface.
"""
from rover_domain_core_gym import RoverDomainGym
import code.ccea_2 as ccea
import code.agent_domain_2 as domain

episodeCount = 1000  # Number of learning episodes

sim = RoverDomainGym()

ccea.initCcea(input_shape=8, num_outputs=2, num_units=32)(sim.data)

for episodeIndex in range(episodeCount):
    sim.data["Episode Index"] = episodeIndex

    obs = sim.reset()
    ccea.assignCceaPolicies(sim.data)

    done = False
    stepCount = 0
    while not done:
        # Select actions and create the joint action from the simulation data
        # Note that this specific function extracts "obs" from the data structure directly, which is why obs is not
        # directly used in this example.
        domain.doAgentProcess(sim.data)
        jointAction = sim.data["Agent Actions"]
        obs, reward, done, info = sim.step(jointAction)
        stepCount += 1

    ccea.rewardCceaPolicies(sim.data)
    ccea.evolveCceaPolicies(sim.data)
    print(sim.data["Global Reward"])
