# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:54:16 2018

@author: Romain Deffayet
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_results(evolution):
    radius = 3
    
    moving_average = [evolution[k][1] for k in range(radius, len(evolution) - radius)]
    
    plt.plot(moving_average)
    plt.show()