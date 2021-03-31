
# In minutes
BOAT_LOAD_TIME = 0.25
DAY_LENGTH = 12*60
BOAT_CAPACITY = 8 # parameter C from the model
MAX_GROUP_SIZE = 7 # C - 1
MAX_LINE_SKIP = 2 # parameter S from the model. Minimum value is 1
COLS_TIMESTEPS = ['time', 'line length','srq length', 'boat occupancy']
COLS_GROUPS = ['size', 'arrival time', 'departure time', 'wait time']
AVERAGE_GROUPS = 1.6 # parameter lambda from the model
DIST1 = [1,1,1,1,1,1,1]
DIST2 = [3,3,2,2,1,1,1]
DIST3 = [1,1,1,2,2,2,2]
GROUP_SIZE_DISTRIBUTION = DIST1

def timesteps_in_day() -> int:
    return int(DAY_LENGTH // BOAT_LOAD_TIME)
