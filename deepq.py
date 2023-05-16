import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'enabled_actions')) # transition tuple

class ReplayBuffer(object):
    def __init__(self, maxlen=100000) -> None:
        self.buffer = deque([], maxlen=maxlen)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def push_mdp_tensor(self, obs, action, reward, _obs, enabled_actions):
        obs = T.tensor(obs, dtype=T.float32).unsqueeze(0)
        action = T.tensor([[action]], dtype=T.long)
        reward = T.tensor([reward], dtype=T.float32)
        _obs = T.tensor(_obs, dtype=T.float32).unsqueeze(0)
        enabled_actions = enabled_actions
        self.push(obs, action, reward, _obs, enabled_actions)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(self, input_dims, fc_dims, num_actions):
        super(DQNetwork, self).__init__()

        # Action space, used to map actions to indices of output layer
        self.num_actions = num_actions # number of actions
        self.input_dims = input_dims # list of input dimensions
        self.fc_dims = fc_dims # list of fully connected layer dimensions

        # Define layers
        self.fc = [nn.Linear(*self.input_dims, self.fc_dims[0])] # first fully connected layer
        for i in range(1, len(self.fc_dims)):
            # Add fully connected layers
            self.fc.append(nn.Linear(self.fc_dims[i-1], self.fc_dims[i]))
        self.fc.append(nn.Linear(self.fc_dims[-1], num_actions)) # output layer
    
    def forward(self, state):
        # Forward pass through network
        x = state
        for layer in self.fc[:-1]:
            x = F.relu(layer(x)) # ReLU activation for all but last layer
        return self.fc[-1](x) # return output of last layer (Q values for all actions)


class DQAgent():
    def __init__(self, gamma, epsilon, alpha, input_dims, num_actions, fc_dims=[256, 256], max_mem_size=100000, batch_size=64, eps_min=.01, eps_dec = .995, opt=max, verbose=False):
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.alpha = alpha # learning rate
        self.input_dims = input_dims # list of input dimensions
        self.num_actions = num_actions # number of actions
        self.fc_dims = fc_dims # list of fully connected layer dimensions
        self.mem_size = max_mem_size # maximum memory size
        self.batch_size = batch_size # batch size
        self.eps_min = eps_min # minimum exploration rate
        self.eps_dec = eps_dec # exploration rate decay
        self.opt = opt # function to determine what is considered optimal, i.e., max or min
        self.torch_opt = T.max if opt == max else T.min # torch equivalent of opt
        self.torch_argopt = T.argmax if opt == max else T.argmin # torch equivalent of argopt
        self.verbose = verbose # whether to print debug information

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device to use for training

        self.policy_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims, num_actions=num_actions).to(self.device) # policy network
        self.target_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims, num_actions=num_actions).to(self.device) # target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy network weights to target network
        self.parameters = nn.ModuleList(self.policy_net.fc).parameters() # list of parameters for all layers
        self.optimizer = optim.AdamW(self.parameters, lr=alpha, amsgrad=True) # AdamW optimizer
        self.buffer = ReplayBuffer(max_mem_size) # replay buffer
        self.loss = nn.SmoothL1Loss() # Huber loss

    def choose_action(self, observation, enabled_actions, force_greedy=False):
        if force_greedy or random.uniform(0, 1) > self.epsilon:
            # Choose action greedily (exploit)
            with T.no_grad():
                state = T.tensor(observation, dtype=T.float32).to(self.device)
                q_values = self.policy_net(state)[enabled_actions] # get Q values for all actions
                action = self.torch_argopt(q_values).item() # choose action with highest Q value
                return enabled_actions[action], q_values[action]
        else:
            # Choose action randomly (explore)
            return np.random.choice(enabled_actions), None
        
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return # don't learn until memory is full
        
        transitions = self.buffer.sample(self.batch_size) # sample batch from memory
        batch = Transition(*zip(*transitions)) # convert batch of transitions to transition of batches
        state_batch = T.cat(batch.state) # concatenate states (n x state_dim)
        action_batch = T.cat(batch.action) # concatenate actions (n x 1)
        reward_batch = T.cat(batch.reward) # concatenate rewards (n x 1)
        next_state_batch = T.cat(batch.next_state) # concatenate next states (n x state_dim)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # get Q values for all actions taken in batch (n x 1)
        with T.no_grad():
            next_state_values = self.torch_opt(self.target_net(next_state_batch), 1)[0] # get optimal Q values of all actions in next state (n x 1)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # calculate expected state action values
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1)) # calculate loss
        self.optimizer.zero_grad() # zero gradients
        loss.backward() # backpropagate loss
        self.optimizer.step() # update parameters
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min # decay exploration rate

        return loss.item() # return loss
