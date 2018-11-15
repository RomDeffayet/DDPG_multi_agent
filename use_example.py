# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:16:11 2018

@author: asus
"""
import numpy as np
import numpy.random as rd

from make_env import make_env 

env = make_env('simple_tag')

states = env.reset()

for i in range (500):
    env.render()
    n = env.n
    action = [rd.random(5) for k in range(n)]
    env.step(action)
    
    