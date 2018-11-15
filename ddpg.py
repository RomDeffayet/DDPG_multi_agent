# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:04:15 2018

@author: Romain Deffayet
"""


class Actor:
    def __init__(self, eval_actions, target_actions, state_space_size, n_actions):
        
    
    def buildNetwork(self, in_size,  N_hidden = 50):
        model = nn.Sequential(
                    nn.Linear(in_size, N_hidden),
                    nn.ReLU(),
                    nn.Linear(N_hidden, self.n_actions),
                    nn.Tanh()
                    nn.Softmax()
                )
        return model
        
        
    def ornsteinUhlenbeck(self, T, mu = [np.zeros(env.action_space[i].n) for i in range(env.n)], 
                                   sigma = 0.3, theta = 0.15, dt = 0.01, x0 = np.zeros_like(mu)):
        m, n = np.size(mu)
        random_process = np.zeros(T, n, m)
        random_process[0] = x0
        for t in range(1,T):
            random_process[t] = random_process[t-1] + theta*(mu - random_process[t-1])*dt + \
            sigma*np.sqrt(dt)*rd.normal(0, 1, (n,m))            
        return random_process
        
        
    def chooseAction(self,state):
        
        
        
    def learn(self,transition, alpha = 0.001 , gamma = 0.9, tau = 0.01):
        
    
       
    
    
class Critic:
    def __init__():
        
        
        
    def buildNetwork():
        
        
        
    def learn():



