from specifics import getSim
from mods import *
from shutil import copyfile
import random

# NOTE: Add the mod functions (variables) to run to modCol here:
modCol = [
    globalRewardSizeCurrMod10,
    globalRewardSizeCurrMod20,
    globalRewardSizeCurrMod30,
    globalRewardSizeCurrMod40,
    globalRewardCoupCurrMod1,
    globalRewardCoupCurrMod2,
    globalRewardCoupCurrMod3,
    globalRewardCoupCurrMod4,
    globalRewardCoupCurrMod5,
    differenceRewardSizeCurrMod10,
    differenceRewardSizeCurrMod20,
    differenceRewardSizeCurrMod30,
    differenceRewardSizeCurrMod40,
    differenceRewardCoupCurrMod1,
    differenceRewardCoupCurrMod2,
    differenceRewardCoupCurrMod3,
    differenceRewardCoupCurrMod4,
    differenceRewardCoupCurrMod5,
    
]

# modCol = [
#     globalRewardMod,
#     differenceRewardMod
# ]

def copyTestFiles():
    copyfile("mods.py", "log/%s/mods.py"%sim.data["Specifics Name"])
    copyfile("specifics.py", "log/%s/specifics.py"%sim.data["Specifics Name"])
    

def main():
    getSim().run()
    
i = 0
while True:
    print("Run %i"%(i))
    random.shuffle(modCol)
    for mod in modCol:
        sim = getSim()
        mod(sim)
        sim.run()
    i += 1

    
# main()
# sim = getSim()
# sim.run()