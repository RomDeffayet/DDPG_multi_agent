# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:28:11 2018

@author: Romain Deffayet
"""

import torch
import torch.nn as nn

dtype = torch.float
device = torch.device("cpu")


def ornsteinUhlenbeck(x_prev, mu = np.zeros_like(x_prev), 
                          sigma = 0.3, theta = 0.15, dt = 0.01):
        m, n = np.size(x_prev)
        x = x_prev + theta*(mu - x_prev)*dt + sigma*np.sqrt(dt)*rd.normal(0, 1, (n,m))            
        return x


def sample(buffer, N):
    if len(buffer) <= N:
        return buffer
    else:
        return rd.choice(buffer, N, replace = False)
           
    
    

def episode(n_episodes = 1000, x0 = 0, buffer_size = render = False):
    
    env = make_env('simple_tag')
    
    
    actors, critics = [], []
    for i in range(env.n):
        actors.append(Actor(env.observation_space[i].shape[0], env.action_space[i].n))
        critics.append(Critic(env.observation_space[i].shape[0], env.action_space[i].n, actors[i]))
    
    replay_buffer = deque()
    
    for ep in range(n_episodes):
            
        noise = x0
        state = env.reset()
        
        ep_rewards = 0
        step_count = 0
        
        while all(done):
            if render:
            env.render()
            
            ###Choose an action and go to next state
            actions = []
            for i in range(env.n):
                noise = ornsteinUhlenbeck(noise)
                actions.append(np.clip(actors[i].forwardPass(state) + noise, -2, 2))
            next_state, rewards, done, _ = env.step(actions)
            rewards = rewards - 500*done
            
            ###Store in the replay buffer
            replay_buffer.append([state, actions, rewards, next_state])
            if len(replay_buffer)>buffer_size:
                replay_buffer.popleft()
                
            ###Sample a minibatch from the buffer
            minibatch = sample(replay_buffer, N)
            
            ###Learn from this minibatch
            for i in range(env.n):
                critics[i].learn(minibatch)
                actors[i].learn(minibatch)
            
            ###Prepare for next step
            step_count +=1
            state = next_state
        
        ep_rewards /= step_count
            
            
        
        
            
    
    
    
    