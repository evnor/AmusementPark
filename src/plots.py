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
    print(data)
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

def plot_stability(data: RunResult, start: int=1, end: int=None) -> None:
    print(data)
    runs, _ = max(data.timesteps.index)
    runs += 1
    if end:
        runs = end
    fig, ax = plt.subplots()
    for n in range(start, runs):
        df_timesteps = data.timesteps.loc[1:n].groupby(by='idx').mean()
        df_timesteps['length filtered'] = gaussian_filter1d(df_timesteps['line length'], sigma=100, mode='constant')
        df_timesteps['length filtered'].plot(ax=ax)
    ax.legend(list(range(start,runs+1)))
    data.timesteps.plot(x='time', y='line length')
    sns.violinplot(data=data.groups, x='size', y='wait time')

def plot_lineskip(data: Tuple[List[RunResult], List[RunResult]]) -> None:
    nonsrq, srq = data
    stats_nonsrq = get_avg_occupancy(nonsrq)
    stats_srq = get_avg_occupancy(srq)
    print(stats_srq-stats_nonsrq)
    print((stats_srq-stats_nonsrq).mean())
    print(stats_nonsrq)
    ax = stats_nonsrq.plot()
    stats_srq.plot(ax=ax)
    ax.legend(['Non-SRQ', 'SRQ'])
    ax.set_xlabel('Max line skip')
    ax.set_ylabel('Mean occupancy')
