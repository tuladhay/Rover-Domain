import csv 
import os
import errno
from parameters import Parameters as p
import numpy as np

    
def save_reward_history(data):
    save_file_name = data["Performance Save File Name"]
    # Create File Directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_file_name)):
        try:
            os.makedirs(os.path.dirname(save_file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(save_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation"] + list(range(p.generations)))
        for s in range(p.stat_runs):
                writer.writerow(['Performance'] + data['Reward History'][s])

def create_reward_history(data):  # Keeps track of ouput from "best" policy each generation
    data["Reward History"] = [[] for i in range(p.stat_runs)]
     
def update_reward_history(data, srun):
    data["Reward History"][srun].append(data["Global Reward"])
        
def print_global_reward(data):
    if data["World Index"] == 0:
        print(data["Global Reward"])
