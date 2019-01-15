import pickle
import os

def savePickle(data):
    save_file_name = data["Pickle Save File Name"]
    
    if not os.path.exists(os.path.dirname(save_file_name)):
        try:
            os.makedirs(os.path.dirname(save_file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    with open(save_file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)