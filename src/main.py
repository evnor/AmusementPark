import constants as const
from classes import RunResult, Group, State
import plots as plt
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import random
import pickle


def generate_groups(time: int, dist: List[int]) -> List[Group]:
    """Generates the groups to be added to the line. 
    Time is the arrival time.
    
    Assumptions:
        len(dist) == const.MAX_GROUP_SIZE - 1
    """
    ngroups = np.random.poisson(lam = const.AVERAGE_GROUPS)
    return generate_n_groups(time, dist, ngroups)

def generate_n_groups(time: int, dist: List[int], ngroups: int) -> List[Group]:
    """Generate ngroups groups according to dist.
    Time is the arrival time
    
    Assumptions:
        len(dist) == const.MAX_GROUP_SIZE - 1
    """
    sizes = random.choices(list(range(1, const.MAX_GROUP_SIZE + 1)), dist, k=ngroups)
    return [Group(n, time) for n in sizes]

def step_time(state: State, time: int, use_srq: bool, infinite: bool=False) -> State:
    """Perform a single timestep, generating arrivals and loading a boat.
    
    Returns the new state (modified in place, not a copy).
    use_srq should be the same value as passed to the constructor of state.
    """
    # Generate the groups
    if not infinite:
        arrivals = generate_groups(time, const.GROUP_SIZE_DISTRIBUTION)
    else:
        arrivals = generate_n_groups(time, const.GROUP_SIZE_DISTRIBUTION, const.BOAT_CAPACITY + const.MAX_LINE_SKIP - len(state.line) - 1)
        
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
    
    Returns a RunResult
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
    """Joins the dataframes of a list of RunResults.
    
    Would be a static method of RunResult, but decorators 
    are outside the scope of 2WH20.
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
    """ Performs nruns runs and joins the results into a single RunResult.
    
    Assumption:
     * nruns, n_timesteps >= 1
    """
    results = []
    for i in range(nruns):
        results.append(perf_timesteps(n_timesteps, use_srq, infinite=infinite))
        print('.', end='')
        if i % 20 == 19:
            pickle_save(results, 'temp')
    return join_runresults(results)

def gather_average_group_data(filename: str) -> Dict[float, RunResult]:
    """Gather the data for do_average_group_plotting. Save the data to filename.pickle.
    """
    const.MAX_LINE_SKIP = 1
    results = {}
    for average_arrivals in range(30, 41):
        const.AVERAGE_GROUPS = average_arrivals / 20
        results[average_arrivals / 20] = perf_n_runs(5, const.timesteps_in_day(), False)
        print(average_arrivals)
    pickle_save(results, filename)
    return results
    
def gather_skip_data(filename: str) -> Tuple[Dict[int, RunResult], Dict[int, RunResult]]:
    """Gather the data for do_lineskip_plotting. Save the data to filename.pickle.
    """
    nonsrq = {}
    srq = {}
    for lineskip in range(1, 40):
        const.MAX_LINE_SKIP = lineskip
        nonsrq[lineskip] = perf_n_runs(15, const.timesteps_in_day(), False, infinite=True)
        print('non-srq done', end='\t')
        srq[lineskip] = perf_n_runs(15, const.timesteps_in_day(), True, infinite=True)
        print('srq done:', lineskip)
        if lineskip % 5 == 0:
            pickle_save((nonsrq, srq), filename)
    pickle_save((nonsrq, srq), filename)
    return (nonsrq, srq)

def gather_stability_condition_confirm_data(filename: str) -> Tuple[RunResult, RunResult]:
    """Gather the data for do_confirm_stability_condition. Save the data to filename.pickle.
    """
    const.MAX_LINE_SKIP = 1
    const.GROUP_SIZE_DISTRIBUTION = const.DIST1
    const.AVERAGE_GROUPS = 1.53
    stable = perf_n_runs(200, 6000, False)
    const.AVERAGE_GROUPS = 1.59
    unstable = perf_n_runs(200, 6000, False)
    pickle_save((stable, unstable), filename)
    return (stable, unstable)
    

def gather_stability_data(filename: str) -> RunResult:
    """Gather the data for do_stability_plotting. Save the data to filename.pickle.
    """
    result = perf_n_runs(500, const.timesteps_in_day(), False)
    pickle_save(result, filename)
    return result

def do_average_group_plotting() -> None:
    """Create a plot of the line length agains lambda (const.AVERAGE_GROUPS)
    
    This was for initial investigations into stability
    """
    # data = gather_average_group_data('average_groups')
    data = pickle_load('average_groups')
    plt.plot_average_groups(data)
    
def do_lineskip_plotting() -> None:
    """Create a plot of the mean occupancy agains maxLineSkip
    
    Figures 4.1, 4.2, 4.3
    """
    # data = gather_skip_data('lineskip5')
    data = pickle_load('lineskip')
    plt.plot_lineskip(data)
    data = pickle_load('lineskip3')
    plt.plot_lineskip(data)
    data = pickle_load('lineskip4')
    plt.plot_lineskip(data)
    
def do_stability_plotting() -> None:
    """Create a plot of the line length for multiple runs at one value of lambda
    
    Figures 4.5, 4.6
    """
    # data = gather_stability_data('500runs_lowerL')
    data = pickle_load('500runs_lowerL')
    plt.plot_stability(data)
    
def do_confirm_stability_condition() -> None:
    """Create a plot of the line length for two values of lambda
    
    Figure 4.4
    """
    # data = gather_stability_condition_confirm_data('confirm_stability_condition4')
    data = pickle_load('confirm_stability_condition4')
    plt.plot_confirm_stability_condition(data)

def do_group_dist_plots() -> None:
    """Create plots of group sizes.
    
    Figures 4.8, 4.9, 4.10
    """
    data1,data2, data3,data4, data5,data6 = pickle_load('group_dist')
    const.MAX_LINE_SKIP = 1
    
    # const.AVERAGE_GROUPS = 1.5
    # const.GROUP_SIZE_DISTRIBUTION = const.DIST1
    # data1 = perf_n_runs(20, const.timesteps_in_day(), False)
    # data2 = perf_n_runs(20, const.timesteps_in_day(), True)
    # pickle_save((data1, data2, data3, data4, data5, data6), 'group_dist')
    
    # const.AVERAGE_GROUPS = 2.05
    # const.GROUP_SIZE_DISTRIBUTION = const.DIST2
    # data3 = perf_n_runs(20, const.timesteps_in_day(), False)
    # data4 = perf_n_runs(20, const.timesteps_in_day(), True)
    # pickle_save((data1, data2, data3, data4, data5, data6), 'group_dist')
    
    # const.AVERAGE_GROUPS = 1.3
    # const.GROUP_SIZE_DISTRIBUTION = const.DIST3
    # data5 = perf_n_runs(20, const.timesteps_in_day(), False)
    # data6 = perf_n_runs(20, const.timesteps_in_day(), True)
    # pickle_save((data1, data2, data3, data4, data5, data6), 'group_dist')
    
    plt.plot_group_dist(data1, data2, ylim=60)
    plt.plot_group_dist(data3, data4, ylim=60)
    plt.plot_group_dist(data5, data6, ylim=60)
    
def do_already_stable_increase_S() -> None:
    """Show behavior when increasing S in an already stable system
    
    Figure 4.7
    """
    const.AVERAGE_GROUPS = 1.45
    const.GROUP_SIZE_DISTRIBUTION = const.DIST1
    # data=[]
    # for S in [1,2,3,4]:
    #     const.MAX_LINE_SKIP = S
    #     data.append(perf_n_runs(100, const.timesteps_in_day(), False))
    # data = tuple(data)
    # pickle_save(data, 'stable_increase_S')
    data = pickle_load('stable_increase_S')
    plt.plot_already_stable_increase_S(data, [1,2,3,4])
    
def do_hist_wait_time() -> None:
    """Create a histogram of the waiting times.
    
    Figure 4.11
    """
    data1,data2, data3,data4, data5,data6 = pickle_load('group_dist')
    plt.plot_hist_wait_time(data1, data2)
    
def get_statistics_of_mean(data: RunResult) -> None:
    """Gather data about standard deviations.
    
    Table 4.2
    """
    means = data.timesteps.groupby(by='run')[['line length', 'boat occupancy']].mean()
    # print(means)
    for n in [10, 20, 50, 100, 200, 500]:
        print(f'n={n}')
        std = means.loc[0:n-1].std()
        print(std)
        print(std / (n**0.5))
    

if __name__ == "__main__":
    # result = perf_timesteps(100, False)
    # pickle_save(result, 'test')
    # result = pickle_load('test')
    # plt.create_plots_single(result)
    
    do_average_group_plotting()
    do_lineskip_plotting()
    do_stability_plotting()
    do_confirm_stability_condition()
    do_already_stable_increase_S()
    do_group_dist_plots()
    do_hist_wait_time()
    get_statistics_of_mean(pickle_load('500runs_lowerL'))
    
    # result = perf_n_runs(5,100,True)
    # print(len(result.timesteps))
    # print(result.timesteps)
    # print(result.groups)
