"""
This is merely a quick and dirty example. It doesn't provide any support for
customisation, extending functionality, or more extensive data collection.
"""
import constants as const
from classes import RunResult, Group, State
import plots as plt
from typing import List
import numpy as np
import pandas as pd
import random
import pickle


def generate_groups(time: int, dist: List[int]) -> List[Group]:
    """Generates the groups to be added to the line.
    Groups are tuples of ints in the form (size, arrival_time).
    """
    ngroups = np.random.poisson(lam = const.AVERAGE_GROUPS)
    sizes = random.choices(list(range(1, const.MAX_GROUP_SIZE + 1)), dist, k=ngroups)
    return [Group(n, time) for n in sizes]

def generate_groups_fancy(time: int, dist: List[int], line_length: int, busyness: float, line_target: int)-> List[Group]:
    """
    Generates the groups to be added to the line.
    Groups are tuples of ints in the form (size, arrival_time).
    This one is more fancy because it takes the distribution and busyness into consideration.
    line_length is the current line length, and the line_target is the length,
    before visitors start to get discouraged by line_length.
    dist is a list of weights, for each group size. For example: [7,6,5,4,3,2,1]
    """
    
    if line_length < line_target:
        groups = random.randint(1,4)
    else:
        if random.uniform(0,1) < busyness:
            groups = random.randint(0,1)
        else:
            groups = 0
    sizes = random.choices(list(range(1, const.MAX_GROUP_SIZE + 1)), dist, k = groups)
    return [Group(n, time) for n in sizes]
        
    
    # Parameters of group size distribution 
    # How busy it is in the park
    # Length of the line 

def step_time(state: State, time: int, use_srq: bool) -> State:
    """Generates new groups and loads a single boat. Returns line, departed_groups
    Looks forward MAX_LINE_SKIP groups to load the boat. (so skipping is possible)
    """

    # People arrive
    arrivals = generate_groups(time, [1,1,1,1,1,1,1])
    # arrivals = generate_groups_fancy(time, [1,1,1,1,1,1,1], 5, 1, 8)
    
    # Sort groups into SRQ, if necessary
    if use_srq:
        for group in arrivals:
            if group.size == 1:
                state.srq.append(group)
            else:
                state.line.append(group)
    else:
        state.line.extend(arrivals)
    
    # Load the boat and keep track of departed groups
    remaining = const.BOAT_CAPACITY
    departed = []
    while len(state.line) > 0:
        options = [index for index, option in enumerate(state.line[0:const.MAX_LINE_SKIP]) if option.size <= remaining]
        if len(options) > 0:
            departed.append(state.line.pop(options[0]))
            remaining -= departed[-1].size
        else:
            break
    
    if use_srq:
        while len(state.srq) > 0 and remaining > 0:
            departed.append(state.srq.pop(0))
            remaining -= 1
            
    state.departed_groups = departed
    
    return state

def perf_timesteps(n: int, use_srq: bool) -> RunResult:
    """Simulate a day of n timesteps. 
    Returns two DataFrames, one containing info about the timesteps, 
    the other containing info about groups
    """
    # Initialize variables to keep track of data
    state = State(use_srq)
    results = RunResult(use_srq)
    
    for time in range(n):
        state = step_time(state, time, use_srq) # Step forward 1 timestep
        
        # Add data to df_timesteps. The time, line length and boat occupancy are tracked
        results.add_timestep(time, state)
        
        # Add data to df_groups. One row per group, keeping track of sizes and times.
        results.add_groups(time, state)
        # print(state)
        state.departed_groups = None
        
    return results

def pickle_save(obj: object, filename: str) -> None:
    """Saves obj to ../pickles/filename.pickle using the pickle library
    """
    with open(f'../pickles/{filename}.pickle', 'wb') as f:
        pickle.dump(obj, f)
        print(f'Saved {filename}.pickle')

def pickle_load(filename: str) -> object:
    """Loads obj from ../pickles/filename.pickle using the pickle library
    """
    with open(f'../pickles/{filename}.pickle', 'rb') as f:
        print(f'Loading {filename}.pickle')
        return pickle.load(f)

def join_runresults(results: List[RunResult]) -> RunResult:
    """Join the dataframes of a list of RunResults.
    """
    timesteps = [res.timesteps for res in results]
    groups = [res.groups for res in results]
    timesteps = pd.concat(timesteps, 
                     keys=list(range(len(results))),
                     names=['run', 'idx']
                     )
    groups = pd.concat(groups,  
                     keys=list(range(len(results))),
                     names=['run', 'idx']
                     )
    joined = RunResult(results[0].use_srq)
    joined.timesteps = timesteps
    joined.groups = groups
    return joined
    

def perf_n_runs(nruns: int, n_timesteps: int, use_srq:bool) -> RunResult:
    """ Performs nruns runs and joins the results.
    
    Assumption:
     * nruns, n_timesteps >= 1
    """
    results = []
    for _ in range(nruns):
        results.append(perf_timesteps(n_timesteps, use_srq))
    return join_runresults(results)

def gather_average_group_data(filename: str):
    const.MAX_LINE_SKIP = 1
    results = {}
    for average_arrivals in range(30, 41):
        const.AVERAGE_GROUPS = average_arrivals / 20
        results[average_arrivals / 20] = perf_n_runs(5, const.timesteps_in_day(), False)
        print(average_arrivals)
    pickle_save(results, filename)

def do_average_group_plotting():
    gather_average_group_data('average_groups')
    data = pickle_load('average_groups')
    plt.plot_average_groups(data)

if __name__ == "__main__":
    result_without_srq = perf_timesteps(200, False)
    # pickle_save(result, 'test')
    # result = pickle_load('test')
    plt.create_plots_single(result_without_srq)
    #do_average_group_plotting()
    
    
    result_with_srq = perf_n_runs(5,200,True)
    # print(len(result.timesteps))
    # print(result.timesteps)
    # print(result.groups)
    plt.plot_compare_srq(result_without_srq, result_with_srq)
