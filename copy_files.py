from specifics import getSim
from mods import *
from shutil import copyfile
import os
import errno

sim = getSim()

saveFolderName = "log/%s"%\
        (sim.data["Specifics Name"])
        
        
try:
    os.makedirs(os.path.dirname(saveFolderName))
except OSError as exc: # Guard against race condition
    if exc.errno != errno.EEXIST:
        raise

copyfile("mods.py", "log/%s/mods.py"%sim.data["Specifics Name"])
copyfile("specifics.py", "log/%s/specifics.py"%sim.data["Specifics Name"])