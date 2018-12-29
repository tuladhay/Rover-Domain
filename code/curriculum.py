"""
Provides coupling and size curriculum functionality. Modifies either "Coupling",
    or "Setup Size". Stores and resets the original (target) domain details with
    a "Original" prefix.

Writes:
"Coupling" (int)
"Original Coupling" (int)
"Setup Size" (double)
"Original Setup Size" (double)

Reads:
"Schedule": (Arraylike of Tuples (coupling/setup size, duration)) Curriculum
    Schedule in number of episodes (generations). e.g. ((1,100), (2, 100), (3, 
    100))
"Episode Index" (int)
"Coupling" (int)
"Original Coupling" (int)
"Setup Size" (double)
"Original Setup Size" (double)
"""

def setCurriculumCoupling(data):
    """
    Set coupling according to curriculum schedule
    """
    schedule = data["Schedule"]
    episodeIndex = data["Episode Index"]
    originalCoupling = data["Coupling"] 
    
    generationSum = 0
    for coupling, duration in schedule:
        generationSum += duration
        trainingCoupling = coupling
        if generationSum > episodeIndex:
            break
            
    data["Original Coupling"] = originalCoupling
    data["Coupling"] = trainingCoupling
            
def restoreCoupling(data):
    """
    Restore coupling to value of coupling before change by the curriculum
    """
    data["Coupling"] = data["Original Coupling"]
    
    
def setCurriculumWorldSize(data):
    """
    Set setup size according to curriculum schedule
    """
    schedule = data["Schedule"]
    episodeIndex = data["Episode Index"]
    originalSetupSize = data['Setup Size']

    generationSum = 0
    for setupSize, duration in schedule:
        generationSum += duration
        trainingSetupSize = setupSize
        if generationSum > episodeIndex:
            break
            
    data["Original Setup Size"] = originalSetupSize
    data['Setup Size'] = trainingSetupSize
            
def restoreWorldSize(data):
    """
    Set setup size according to curriculum schedule
    """
    data['Setup Size'] = data["Original Setup Size"] 