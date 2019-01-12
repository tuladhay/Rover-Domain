import csv 
import os
import errno

    
def save_reward_history(data):
    saveFileName  = data["Performance Save File Name"]
    # Create File Directory if it doesn't exist
    if not os.path.exists(os.path.dirname(saveFileName)):
        try:
            os.makedirs(os.path.dirname(saveFileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    with open(saveFileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["*Episode"] + list(range(len(data["Reward History"]))))
        writer.writerow(['Performance'] + data["Reward History"])

def create_reward_history(data):
    data["Reward History"] = []
     
def update_reward_history(data):
    data["Reward History"].append(data["Global Reward"])
        
def print_global_reward(data):
    if data["World Index"] == 0:
        print(data["Global Reward"])
                