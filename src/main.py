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
    return generate_n_groups(time, dist, ngroups)

def generate_n_groups(time: int, dist: List[int], ngroups: int) -> List[Group]:
    sizes = random.choices(list(range(1, const.MAX_GROUP_SIZE + 1)), dist, k=ngroups)
    return [Group(n, time) for n in sizes]

def step_time(state: State, time: int, use_srq: bool, infinite: bool=False) -> State:
    """Generates new groups and loads a single boat. Returns line, departed_groups
    Looks forward MAX_LINE_SKIP groups to load the boat. (so skipping is possible)
    """

    if not infinite:
        # People arrive
        arrivals = generate_groups(time, [1,1,1,1,1,1,1])
    else:
        arrivals = generate_n_groups(time, [1,1,1,1,1,1,1], 7 + const.lineskip - len(state.line))
        
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

def perf_timesteps(n: int, use_srq: bool, infinite: bool=False) -> RunResult:
    """Simulate a day of n timesteps. 
    Returns two DataFrames, one containing info about the timesteps, 
    the other containing info about groups
    """
    # Initialize variables to keep track of data
    state = State(use_srq)
    results = RunResult(use_srq)
    
    for time in range(n):
        state = step_time(state, time, use_srq, infinite=infinite) # Step forward 1 timestep
        
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
    

def perf_n_runs(nruns: int, n_timesteps: int, use_srq:bool, infinite: bool=False) -> RunResult:
    """ Performs nruns runs and joins the results.
    
    Assumption:
     * nruns, n_timesteps >= 1
    """
    results = []
    for _ in range(nruns):
        results.append(perf_timesteps(n_timesteps, use_srq, infinite=infinite))
    return join_runresults(results)

def gather_average_group_data(filename: str):
    const.MAX_LINE_SKIP = 1
    results = {}
    for average_arrivals in range(30, 41):
        const.AVERAGE_GROUPS = average_arrivals / 20
        results[average_arrivals / 20] = perf_n_runs(5, const.timesteps_in_day(), False)
        print(average_arrivals)
    pickle_save(results, filename)
    
def gather_skip_data(filename: str):
    nonsrq = {}
    srq = {}
    for lineskip in range(6, 11):
        const.MAX_LINE_SKIP = lineskip
        nonsrq[lineskip] = perf_n_runs(5, const.timesteps_in_day(), False, infinite=True)
        srq[lineskip] = perf_n_runs(5, const.timesteps_in_day(), True, infinite=True)
        print(lineskip)
    pickle_save((nonsrq, srq), filename)

def do_average_group_plotting():
    gather_average_group_data('average_groups')
    data = pickle_load('average_groups')
    plt.plot_average_groups(data)
    
def do_lineskip_plotting():
    gather_skip_data('lineskip_from_6')
    data = pickle_load('lineskip')
    plt.plot_lineskip(data)

if __name__ == "__main__":
    # result = perf_timesteps(100, False)
    # pickle_save(result, 'test')
    # result = pickle_load('test')
    # plt.create_plots_single(result)
    # do_average_group_plotting()
    do_lineskip_plotting()
    
    # result = perf_n_runs(5,100,True)
    # print(len(result.timesteps))
    # print(result.timesteps)
    # print(result.groups)