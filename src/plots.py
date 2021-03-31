from classes import *
import numpy as np
import pandas as pd
# =============================================================================
# %matplotlib inline
# =============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  
from typing import Tuple
sns.set()
plt.rcParams['figure.figsize'] = 10, 5  
plt.rcParams['lines.markeredgewidth'] = 1  
from sklearn.linear_model import LinearRegression  
from sklearn.cluster import KMeans  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from scipy.ndimage.filters import gaussian_filter1d


def create_plots_single(run: RunResult) -> None:
    
    # #Line chart that shows the line length vs the time of day ie busiest times 
    fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, sharex=False, sharey=False, figsize=(20,5))
    
    run.timesteps['line length'].plot(color='orange', ax=ax[0,0])
    ax[0,0].set_title('Variation in Line Length by Time of Day')
    ax[0,0].set_xlabel('Timesteps')
    ax[0,0].set_ylabel('Line Length');
    
    #Histogram of the waiting times
    ax[0,1] = sns.distplot(run.groups['wait time'], bins=20, color='b', ax=ax[0, 1])
    ax[0,1].set_title("Histogram of the waiting time")
    ax[0,1].set_xlabel("Timesteps")
    ax[0,1].set_ylabel("Wait Time");
    
    # #Histogram of the queue lengths
    ax[0,2] = sns.distplot(run.timesteps['line length'], bins=20, color='r', ax=ax[0, 2])
    ax[0,2].set_title("Histogram of the line length")
    ax[0,2].set_xlabel("Timesteps")
    ax[0,2].set_ylabel("Line length");
    

def plot_compare_srq(normalRun: RunResult, srqRun: RunResult) -> None:
    # When we have an SRQ: (plot describing boat occupancy of SRQ vs NoSRQ)
    fig, ax = plt.subplots(ncols=2, squeeze=False, sharex=False, sharey=True, figsize=(5,7))
    sns.violinplot(data=srqRun.timesteps, x='time', y='occupancy', ax=ax[0,0])
    sns.violinplot(data=normalRun.timesteps, x='time', y='occupancy', ax=ax[0,1])
    ax[0, 0].set_title('Boat Occupancy when Single Rider Queue is Active')
    ax[0, 1].set_title('Boat Occupancy without a Single Rider Queue')
    ax[0, 0].set_xlabel('Timesteps')
    ax[0, 1].set_xlabel('Timesteps')
    ax[0, 0].set_ylabel('Boat Occupancy')
    ax[0, 1].yaxis.set_visible(False);
    
    #box plot comparing wait time of groups for SRQ vs. nonSRQ
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,5))
    sns.boxplot(data=srqRun.groups, x='size', y='wait time', ax=ax[0]) #SRQ
    sns.boxplot(data=normalRun.groups, x='size' , y='wait time', ax=ax[1]) #nonSRQ
    ax[0,0].set_title('Wait time (min) per group when SRQ is active')
    ax[0,1].set_xlabel('Wait time (min) per group when SRQ is not active')
    ax[0,0].set_ylabel('Wait Time (min)')
    ax[0,0].set_xlabel('Group Size')
    ax[0,1].set_xlabel('Group Size');

def get_performance_stats_ungrouped(data: List[RunResult]) -> pd.DataFrame:
    groups = [result.groups for avg, result in data.items()]
    grouped = [df_groups['wait time'].mean() for df_groups in groups]
    df = pd.DataFrame(data=grouped, columns=['mean wait time'], index=data.keys())
    return df

def get_avg_occupancy(data: List[RunResult]) -> pd.DataFrame:
    # print(data)
    timesteps = [result.timesteps for avg, result in data.items()]
    grouped = [df_timesteps['boat occupancy'].mean() for df_timesteps in timesteps]
    df = pd.DataFrame(data=grouped, columns=['mean occupancy'], index=data.keys())
    return df

def plot_average_groups(data: List[RunResult]) -> None:
    timesteps = [res.timesteps for res in data.values()]
    groups = [res.groups for res in data.values()]
    timesteps = pd.concat(timesteps, 
                     keys=data.keys(),
                     names=['avg arrivals', 'run', 'idx']
                     )
    groups = pd.concat(groups,  
                     keys=data.keys(),
                     names=['avg arrivals', 'run', 'idx']
                     )
    timesteps.groupby(by='avg arrivals').plot(x='time', y='line length')

def plot_stability(data: RunResult, start: int=0, end: int=None, milestones: List[int]=[10, 20, 50, 100, 200, 500]) -> None:
    # print(data)
    runs, _ = max(data.timesteps.index)
    runs += 1
    if end:
        runs = end
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    for n in range(start, runs):
        if n+1 in milestones:
            # sns.lineplot(data=gaussian_filter1d(data.timesteps['line length'], sigma=100, mode='constant'))
            df_timesteps = pd.DataFrame(columns=['filtered'], data=gaussian_filter1d(data.timesteps.loc[0:n].groupby(by='idx')['line length'].median(), sigma=100, mode='nearest'))
            df_timesteps['filtered'].plot(ax=ax, x='time', y='line length', legend=False)
        data.timesteps.loc[n].plot(ax=ax2, x='time', y='line length', alpha=0.05, legend=False)
        # df_timesteps['length filtered'] = gaussian_filter1d(df_timesteps['line length'], sigma=100, mode='constant')
    ax.legend(milestones)
    ax.set_xlabel('Time')
    ax2.set_xlabel('Time')
    ax.set_ylabel('Median queue length (smoothed)')
    ax2.set_ylabel('Queue length')
    # data.timesteps.plot(x='time', y='line length')
    # sns.violinplot(data=data.groups, x='size', y='wait time')

def plot_lineskip(data: Tuple[List[RunResult], List[RunResult]]) -> None:
    nonsrq, srq = data
    stats_nonsrq = get_avg_occupancy(nonsrq)
    stats_srq = get_avg_occupancy(srq)
    # print((stats_srq-stats_nonsrq).mean())
    print(stats_nonsrq)
    ax = stats_nonsrq.plot()
    stats_srq.plot(ax=ax, figsize=(10,3))
    ax.legend(['Non-SRQ', 'SRQ'])
    ax.set_xlabel('Max line skip')
    ax.set_ylabel('Mean occupancy')
    ax.set_ylim(6, 8.05)

def plot_confirm_stability_condition(data: Tuple[RunResult, RunResult]) -> None:
    stable, unstable = data
    ax = stable.timesteps.groupby(by='time').mean().plot(y='line length')
    unstable.timesteps.groupby(by='time').mean().plot(y='line length', ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean line length')
    ax.legend(['λ=1.53', 'λ=1.59'])

def plot_already_stable_increase_S(data: Tuple[RunResult, ...], Svals: List[int]) -> None:
    fig, ax = plt.subplots()
    for result in data:
        df_timesteps = pd.DataFrame(columns=['filtered'], data=gaussian_filter1d(result.timesteps.groupby(by='time')['line length'].median(), sigma=100, mode='nearest'))
        df_timesteps.plot(ax=ax, y='filtered', legend=False)
        print(result.timesteps['boat occupancy'].mean())
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean line length')
    ax.legend([f'S = {S}' for S in Svals])
    
def plot_group_dist(data1: RunResult, data2: RunResult, ylim: int=40) -> None:
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10,5))
    data1.groups['size'] = data1.groups['size'].astype(int) 
    data2.groups['size'] = data2.groups['size'].astype(int) 
    sns.boxplot(data=data1.groups, x='size', y='wait time', ax=ax[0])
    sns.boxplot(data=data2.groups, x='size', y='wait time', ax=ax[1])
    ax[0].set_ylim(-5,ylim)
    plt.subplots_adjust(wspace=0.1)
    ax[0].set_ylabel('Wait time')
    ax[1].set_ylabel('')
    ax[0].set_xlabel('Group size')
    ax[1].set_xlabel('Group size')
    ax[0].set_title('Without SRQ')
    ax[1].set_title('With SRQ')
    
def plot_hist_wait_time(data1: RunResult, data2: RunResult) -> None:
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10,5))
    #Histogram of the waiting times
    ax[0] = sns.distplot(data1.groups['wait time'], bins=range(0, 110, 8), color='b', ax=ax[0], norm_hist=True, kde=False)
    ax[0].set_title("Without SRQ")
    ax[0].set_xlabel("Wait time")
    ax[0].set_ylabel("Density");
    
    # #Histogram of the queue lengths
    ax[1] = sns.distplot(data2.groups['wait time'], bins=range(0, 110, 8), color='b', ax=ax[1], norm_hist=True, kde=False)
    ax[1].set_title("With SRQ")
    ax[1].set_xlabel("Timesteps")
    ax[1].set_ylabel("Wait Time");
    plt.subplots_adjust(wspace=0.1)