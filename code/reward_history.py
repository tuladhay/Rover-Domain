import csv 
import os
import errno
from parameters import Parameters as p

    
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
        writer.writerow(["Stat Run"] + list(range(len(data["Reward History"]))))
        writer.writerow(['Performance'] + data["Reward History"])

def create_reward_history(data):  # Keeps track of ouput from "best" policy each generation
    data["Reward History"] = [[0.0 for i in range(p.generations)] for j in range(p.stat_runs)]
     
def update_reward_history(data, gen, stat_run):
    data["Reward History"][stat_run][gen] = data["Global Reward"]
        
def print_global_reward(data):
    if data["World Index"] == 0:
        print(data["Global Reward"])
                