import datetime
from code.reward import * # Agent Reward
from code.curriculum import * # Agent Curriculum


def globalRewardMod(sim):
    sim.data["Mod Name"] = "global"
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardMod(sim):
    sim.data["Mod Name"] = "difference"
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)

def dppRewardMod(sim):
    sim.data["Mod Name"] = "dpp"
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignDppReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)



def globalRewardSizeCurrMod10(sim):
    sim.data["Schedule"] = ((10.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr10"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardSizeCurrMod20(sim):
    sim.data["Schedule"] = ((20.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr20"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        
def globalRewardSizeCurrMod30(sim):
    sim.data["Schedule"] = ((30.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr30"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardSizeCurrMod40(sim):
    sim.data["Schedule"] = ((40.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "globalSizeCurr40"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] =assignGlobalReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        

def globalRewardCoupCurrMod1(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((1, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr1"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod2(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((2, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr2"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod3(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((3, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr3"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod4(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((4, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr4"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def globalRewardCoupCurrMod5(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((5, 2000), (6, 3000))
    sim.data["Mod Name"] = "globalCoupCurr5"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignGlobalReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
##################################################################################


        
def differenceRewardSizeCurrMod10(sim):
    sim.data["Schedule"] = ((10.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr10"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardSizeCurrMod20(sim):
    sim.data["Schedule"] = ((20.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr20"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        
def differenceRewardSizeCurrMod30(sim):
    sim.data["Schedule"] = ((30.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr30"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardSizeCurrMod40(sim):
    sim.data["Schedule"] = ((40.0, 2000), (50.0,3000))
    sim.data["Mod Name"] = "differenceSizeCurr40"
    sim.trainBeginFuncCol.insert(0, setCurriculumWorldSize)
    sim.testBeginFuncCol.insert(0, restoreWorldSize)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
        

def differenceRewardCoupCurrMod1(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((1, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr1"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod2(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((2, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr2"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod3(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((3, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr3"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod4(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((4, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr4"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
def differenceRewardCoupCurrMod5(sim):
    sim.data["Schedule"] = sim.data["Schedule"] = ((5, 2000), (6, 3000))
    sim.data["Mod Name"] = "differenceCoupCurr5"
    sim.trainBeginFuncCol.insert(0, setCurriculumCoupling)
    sim.testBeginFuncCol.insert(0, restoreCoupling)
    
    dateTimeString = datetime.datetime.now().strftime("%m_%d_%Y %H_%M_%S_%f")
    print("Starting %s test at\n\t%s\n"%(sim.data["Mod Name"], dateTimeString))
    
    # Agent Reward 
    sim.data["Reward Function"] = assignDifferenceReward 
    
    
    sim.data["Performance Save File Name"] = "log/%s/%s/performance/perf %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Trajectory Save File Name"] = "log/%s/%s/trajectory/traj %s.csv"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)
        
    sim.data["Pickle Save File Name"] = "log/%s/%s/pickle/data %s.pickle"%\
        (sim.data["Specifics Name"], sim.data["Mod Name"], dateTimeString)