"""
Records test performance and saves it to save file
Saving dDoes not work well with Amazon AWS and other platforms. Might not be 
    compatible with user's operating system.

Writes:
save file to "Performance Save File Name"
"Performance History" (ArrayLike<double>)

Reads:
"Performance History" (ArrayLike<double>)
"""

import csv 
import os
import errno

    
def savePerformanceHistory(data):
    """
    Save performance history to save file
    """
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
        writer.writerow(["*Episode"] + list(range(len(data["Performance History"]))))
        writer.writerow(['Performance'] + data["Performance History"])

def createPerformanceHistory(data):
    data["Performance History"] = []
     
def updatePerformanceHistory(data):
    data["Performance History"].append(data["Performance"])
        
