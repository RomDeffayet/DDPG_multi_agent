# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:28:11 2018

@author: Romain Deffayet
"""

import numpy as np
import numpy.random as rd
import argparse
from collections import deque
import pickle
import os

from ddpg import Actor, Critic
from make_env import make_env

import torch

dtype = torch.float
device = torch.device("cuda")


def ornsteinUhlenbeck(x_prev, mu, 
                          sigma = 0.3, theta = 0.15, dt = 0.01):
        mu = np.zeros_like(x_prev)
        n = np.size(x_prev)
        x = x_prev + theta*(mu - x_prev)*dt + sigma*np.sqrt(dt)*rd.normal(0, 1, n)            
        return x


def sample(buffer, N):
    if len(buffer) <= N:
        return buffer
    else:
        idx = rd.choice(len(buffer), N, replace = False)
        sample = []
        for i in range(N):
            sample.append(buffer[idx[i]])
        return sample
           
    
    

def episode(n_episodes, buffer_size, N, learn, render, x0, mu, sigma, theta, dt,
            alpha, gamma, tau, init_actors = None, init_critics = None):   
    actors, critics = [], []
    for i in range(env.n):
        if init_actors is not None:
            actors = init_actors
            critics = init_critics
        else:
            actors.append(Actor(env.observation_space[i].shape[0], env.action_space[i].n))
            critics.append(Critic(env.observation_space[i].shape[0], env.action_space[i].n, actors[i]))
    
    replay_buffer = deque()
    
    evolution = []
    
    for ep in range(n_episodes):
            
        noise = x0
        state = env.reset()
        
        ep_rewards = np.zeros(env.n)
        step_count = 0
        done = np.array([False] * 4)
        
        while (not any(done) and step_count < 1000):
            if render:
                env.render()
            
            ###Choose an action and go to next state
            actions = []
            for i in range(env.n):
                noise = ornsteinUhlenbeck(noise, mu, sigma, theta, dt)
                action = actors[i].forwardPass(state[i]).detach().numpy()
                actions.append(np.clip(action + noise, -2, 2))
            next_state, rewards, done, _ = env.step(actions)
            rewards = np.asarray(rewards) - 500*np.asarray(done)
            ep_rewards += rewards
            
            if learn:
                ###Store in the replay buffer
                replay_buffer.append(np.array([state, actions, rewards, next_state]))
                if len(replay_buffer)>buffer_size:
                    replay_buffer.popleft()
                    
                ###Sample a minibatch from the buffer
                minibatch = sample(replay_buffer, N)
                
                ###Learn from this minibatch
                for i in range(env.n):
                    critics[i].learn(minibatch, i)
                    actors[i].learn(minibatch, i)
            
            ###Prepare for next step
            step_count +=1
            state = next_state
        
        ep_rewards /= step_count
        print("Episode " + str(ep) + " : " + str(ep_rewards) + " in " + str(step_count) + " steps")
        evolution.append((ep_rewards, step_count))
    return actors, critics, evolution
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str) 
    parser.add_argument('--n_episodes', default=5000, type=int)
    parser.add_argument ('--learn', default=True, type=bool)
    parser.add_argument ('--render', default=False, type=bool)
    parser.add_argument ('--buffer_size', default=1000, type=int)
    parser.add_argument ('--minibatch_size', default=32, type=int)
    parser.add_argument ('--alpha', default=0.001, type=float)
    parser.add_argument ('--gamma', default=0.9, type=float)
    parser.add_argument ('--tau', default=0.01, type=float)
    parser.add_argument ('--ou_x0', default=0, type=float)
    parser.add_argument ('--ou_mu', default=0, type=float)
    parser.add_argument ('--ou_sigma', default=0.3, type=float)
    parser.add_argument ('--ou_theta', default=0.15, type=float)
    parser.add_argument ('--ou_dt', default=0.01, type=float)
    args = parser.parse_args()
    
    env = make_env(args.env)
    
    actors, critics, evolution = episode(n_episodes = args.n_episodes, 
                                                buffer_size = args.buffer_size,
            N = args.minibatch_size, learn = args.learn, render = args.render,
            x0 = args.ou_x0 * np.ones(env.action_space[0].n),
            mu = args.ou_mu * np.ones(env.action_space[0].n),
            sigma = args.ou_sigma, theta = args.ou_theta, dt = args.ou_dt,
            alpha = args.alpha, gamma = args.gamma, tau = args.tau)
    
    pickle.dump(actors, open('actors','wb'))
    pickle.dump(critics, open('critics','wb'))
    pickle.dump(evolution, open('evolution','wb'))
    print(os.getcwd())
    
    
    
        
        
            
    
    
    
    