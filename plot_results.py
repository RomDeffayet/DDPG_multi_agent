# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:54:16 2018

@author: Romain Deffayet
"""

import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(evolution, radius):
    
    steps = [episode[1] for episode in evolution]
    
    average = moving_average(steps, radius)
    
    plt.plot(average)
    plt.xlabel("Number of episodes")
    plt.ylabel("Length of the episode")
    plt.title("Moving average of the length of episode while learning")
    plt.show()