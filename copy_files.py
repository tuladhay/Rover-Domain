from parameters import Parameters as p
from shutil import copyfile
import os
import errno

save_folder_name = "log/%s"%\
        (p.data["Specifics Name"])
        
        
try:
    os.makedirs(os.path.dirname(save_folder_name))
except OSError as exc: # Guard against race condition
    if exc.errno != errno.EEXIST:
        raise

copyfile("mods.py", "log/%s/mods.py"%p.data["Specifics Name"])
copyfile("specifics.py", "log/%s/specifics.py"%p.data["Specifics Name"])