from classes import *
import numpy as np
import pandas as pd
# =============================================================================
# %matplotlib inline
# =============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  
sns.set()
plt.rcParams['figure.figsize'] = 10, 5  
plt.rcParams['lines.markeredgewidth'] = 1  
from sklearn.linear_model import LinearRegression  
from sklearn.cluster import KMeans  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
# I don't know if these are needed but just in case we have them


def create_plots_single(run: RunResult):
    
    #Line chart that shows the line length vs the time of day ie busiest times 
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, sharex=True, sharey=True, figsize=(5,5))
    run.timesteps['line length'].plot(ax=ax[0,0])
    ax[0,0].set_title('Variation in Line Length by Time of Day')
    ax[0,0].set_xlabel('Timesteps')
    ax[0,0].set_ylabel('Line Length');
    

def plot_compare_srq(normalRun: RunResult, srqRun: RunResult):
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
