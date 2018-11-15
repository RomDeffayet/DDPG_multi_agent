# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:04:15 2018

@author: Romain Deffayet
"""
from torch.autograd import Variable

class Critic_Model(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(Critic_Model, self).__init__()
        self.h1 = nn.Linear(state_space_size, 50)
        self.h2 = nn.Linear(50,1)
        self.W1 = nn.Parameter(nn.init.normal_(torch.empty(50,50)))
        self.W2 = nn.Parameter(nn.init.normal_(torch.empty(action_space_size, 50)))
        self.b = nn.Parameter(nn.init.normal_(torch.empty(1,50)))

    def forward(self, s, a):  
        x = F.relu(torch.mm(self.h1(s), self.W1) + torch.mm(a, self.W2) + self.b)
        x = self.h2(x)
        return x



class Actor:
    def __init__(self, state_space_size, n_actions, learning_rate = 0.001):
        
        self.eval_actions = self.buildNetwork(self, state_space_size, n_actions)
        self.target_actions = self.buildNetwork(self, state_space_size, n_actions)
        
        self.optimizer = torch.optim.Adam(self.eval_actions.parameters(), learning_rate)
        
        ###Applying the same weights and biases
        for target_params, eval_params in zip(self.target_actions.parameters(), 
                                              self.eval_actions.parameters()):
            target_params.data.copy_(eval_params.data)
    
    def buildNetwork(self, in_size, n_actions,  N_hidden = 50):
        model = nn.Sequential(
                    nn.Linear(in_size, N_hidden),
                    nn.ReLU(),
                    nn.Linear(N_hidden, n_actions),
                    nn.Tanh()
                )
        return model
    
    def setCritic(self, critic):
        self.critic = critic
        
        
    def forwardPass(self,state):
        return self.eval_actions(state)        
        
        
    def learn(self, minibatch, alpha = 0.001 , tau = 0.01):
        s,a,r,sp = minibatch
        s = Variable(torch.from_numpy(s))
        a = Variable(torch.from_numpy(a)) 
        
        ###Optimizing
        pred_actions = self.forwardPass(self, s)
        loss = -1 * torch.sum(self.critic.forwardPass(s, pred_actions))
        self.optimizer.no_grad()
        loss.backward()
        self.optimizer.step()
        
        ###Updating the target network
        for target_params, eval_params in zip(self.target_actions.parameters(), 
                                              self.eval_actions.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - tau) + eval_params.data * tau)   
    
       
    
    
class Critic:
    def __init__(self, state_space_size, action_space_size, actor, learning_rate = 0.001):
        self.eval_Q = self.buildNetwork(self, state_space_size, action_space_size)
        self.target_Q = self.buildNetwork(self, state_space_size, action_space_size)
        
        self.actor = actor
        self.actor.setCritic(self)
        
        self.optimizer = torch.optim.Adam(self.eval_Q.parameters(), learning_rate)
        
        ###Applying the same weights and biases
        for target_params, eval_params in zip(self.target_Q.parameters(), 
                                              self.eval_Q.parameters()):
            target_params.data.copy_(eval_params.data)
        
        
    def buildNetwork(self, state_space_size, action_space_size):
        return Critic_Model(state_space_size, action_space_size)
        
        
    def learn(self, minibatch, gamma = .9, tau = 0.01):
        s,a,r,sp = minibatch
        s = Variable(torch.from_numpy(s))
        a = Variable(torch.from_numpy(a))
        r = Variable(torch.from_numpy(r))
        sp = Variable(torch.from_numpy(sp))
        
        ### Computing loss
        a_pred = self.actor.target_actions(sp)
        y_pred = r + gamma * self.target_Q(sp, a_pred)
        y_eval = self.eval_Q(s,a)
        loss = F.mse_loss(y_pred, y_eval)
        
        ### Optimizing
        self.optimizer.no_grad()
        loss.backward()
        self.optimizer.step()


        ###Updating the target network
        for target_params, eval_params in zip(self.target_Q.parameters(), 
                                              self.eval_Q.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - tau) + eval_params.data * tau)   
        


