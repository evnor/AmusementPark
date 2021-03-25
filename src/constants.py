
# In minutes
BOAT_LOAD_TIME = 0.25
DAY_LENGTH = 12*60
BOAT_CAPACITY = 8
MAX_GROUP_SIZE = 7
MAX_LINE_SKIP = 4 # 1 is the minimum
COLS_TIMESTEPS = ['time', 'line length','srq length', 'boat occupancy']
COLS_GROUPS = ['size', 'arrival time', 'departure time', 'wait time']
AVERAGE_GROUPS = 1.55
DIST1 = [1,1,1,1,1,1,1]
GROUP_SIZE_DISTRIBUTION = DIST1

def timesteps_in_day() -> int:
    return int(DAY_LENGTH // BOAT_LOAD_TIME)
