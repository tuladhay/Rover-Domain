import rover_domain
import numpy as np


class SequentialPOIRD(rover_domain.RoverDomain):
    """
    Extends the rover domain class to include funcions for sequential POI observation tasks,
    where the reward is not handed out until all types of a POI are observed, and there is a specific
    sequential relationship for how each POI can be observed.
    """
    def __init__(self):
        super(SequentialPOIRD, self).__init__()
        self.poi_types = ['C', 'A', 'B']
        self.n_pois = len(self.poi_types)

        # store the sequence as a graph, with each type pointing to the immediate parent(s) which must be satisfied
        # for it to be counted as a score.
        self.sequence = {"A": None, "B": ["A"], "C": ["B"]}

        # POI visited is a dictionary which keeps track of which types have been visited
        self.poi_visited = {}
        for t in self.sequence:
            self.poi_visited[t] = False

    def reset(self):
        """
        Resets the "poi_visited" dictionary to entirely false, in addition to the parent reset behavior.
        :return: None
        """
        super(SequentialPOIRD, self).reset()
        for t in self.poi_types:
            self.poi_visited[t] = False

    def update_POI_observation(self, poi_type):
        """
        Updates the self.poi_visited[poi_type] if the dependencies are properly met
        Only does the update for the type being queried.

        :param poi_type: String, the type identifier of the poi being observed
        :return: None
        """
        parents = self.sequence[poi_type]
        if parents is not None:
            observed = True
            for p in parents:
                if not self.poi_visited[p]:
                    observed = False
                    break
            if observed:
                self.poi_visited[poi_type] = True
        else:
            # If no parents
            self.poi_visited[poi_type] = True

    def update_sequence_visits(self):
        """
        Uses current rover positions to update observations.
        :return:
        """
        # TODO implement tight coupling in this scenario (can use parent class functionality?)
        for rover in self.rover_positions:
            for i, poi in enumerate(self.poi_positions):
                # POI is within observation distance
                # print("Distance to {} : {}".format(np.array(poi), np.linalg.norm(np.array(rover) - np.array(poi))))
                if np.linalg.norm(np.array(rover) - np.array(poi)) < self.interaction_dist:
                    self.update_POI_observation(self.poi_types[i])

    def sequential_score(self):
        """
        Checks if all the POI in poi_visited have been observed. Returns a fixed score at the moment.
        :return: Global reward score for the agents based on the sequential observation task
        """
        # Update to see if the requirements are now set
        # (done until no more updates are found, to account for cascading updates)
        # Now with the updated graph, we check if all POI are observed
        if all(self.poi_visited.values()):
            return 1
        else:
            return 0


if __name__ == '__main__':
    # Fix random seeds
    import random
    random.seed(0)
    np.random.seed(0)

    rd = SequentialPOIRD()
    #rd.reset()
    print(rd.poi_visited)
    print("Score: ", rd.sequential_score())
    print(np.array(rd.rover_positions))
    print(np.array(rd.poi_positions))
    print()
    rd.interaction_dist = 1
    rd.update_sequence_visits()
    print(rd.poi_visited)
    print("Score: ", rd.sequential_score())
    print()
    rd.interaction_dist = 2
    rd.update_sequence_visits()
    print(rd.poi_visited)
    print("Score: ", rd.sequential_score())
    print()
    rd.interaction_dist = 3
    rd.update_sequence_visits()
    print(rd.poi_visited)
    print("Score: ", rd.sequential_score())

