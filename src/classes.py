import constants as const
import pandas as pd
from typing import List

class Group:
    
    def __init__(self, size: int, arrival_time: int):
        self.size = size
        self.arrival_time = arrival_time
    
    size: int
    arrival_time: int

    def __repr__(self) -> str:
        return "Group({}, {})".format(self.size, self.arrival_time)

    def __str__(self) -> str:
        return "Group({}, {})".format(self.size, self.arrival_time)

        
class State:
    def __init__(self, use_srq):
        self.line = []
        self.srq = []
        
        
    line: List[Group]
    srq: List[Group]
    departed_groups: List[Group]
    
    def __repr__(self) -> str:
        return "State({}, {}, {})".format(self.line.repr(), self.srq.repr(), self.departed_groups.repr())

    def __str__(self) -> str:
        return "Line: {}, SRQ: {}, Departed: {}".format(len(self.line), len(self.srq), len(self.departed_groups))
  
    
class RunResult:
    
    def __init__(self, use_srq: bool):
        self.timesteps = pd.DataFrame(columns = const.COLS_TIMESTEPS, dtype='float')
        self.timesteps.set_index('time')
        self.groups = pd.DataFrame(columns = const.COLS_GROUPS, dtype='float')
        self.use_srq = True
    
    timesteps: pd.DataFrame
    groups: pd.DataFrame
    use_srq: bool
    
    def add_timestep(self, time: int, state: State):
        timestep_row = {'time': time, 
                   'line length': len(state.line), 
                   'boat occupancy': sum(t.size for t in state.departed_groups)
                   }
        if self.use_srq:
            timestep_row['srq length'] = len(state.srq)
        self.timesteps = self.timesteps.append(timestep_row, ignore_index=True)
        
    def add_groups(self, time: int, state: State):
        group_rows = []
        for group in state.departed_groups:
            group_rows.append({'size': group.size,
                               'arrival time': group.arrival_time,
                               'departure time': time,
                               'wait time': time - group.arrival_time
                })
        self.groups = self.groups.append(group_rows, ignore_index=True)