import csv 
import os
import errno

    
def save_reward_history(data):
    save_file_name  = data["Performance Save File Name"]
    # Create File Directory if it doesn't exist
    # if not os.path.exists(os.path.dirname(save_file_name)):
    #     try:
    #         os.makedirs(os.path.dirname(save_file_name))
    #     except OSError as exc: # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    #
    # with open(save_file_name, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["*Episode"] + list(range(len(data["Reward History"]))))
    #     writer.writerow(['Performance'] + data["Reward History"])

def create_reward_history(data):
    data["Reward History"] = []
     
def update_reward_history(data):
    data["Reward History"].append(data["Global Reward"])
        
def print_global_reward(data):
    if data["World Index"] == 0:
        print(data["Global Reward"])
                