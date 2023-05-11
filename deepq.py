import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, action_space):
        super(DQNetwork, self).__init__()
        # Action space, used to map actions to indices of output layer
        self.action_space = action_space # dict of {action: index}
        self.input_dims = input_dims # list of input dimensions
        self.fc1_dims = fc1_dims # number of neurons in first hidden layer
        self.fc2_dims = fc2_dims # number of neurons in second hidden layer

        # Define layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, len(action_space))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # Adam optimizer
        self.loss = nn.MSELoss() # Mean Squared Error Loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # GPU or CPU
        self.to(self.device)

    def forward(self, state):
        # Forward pass through network
        x = F.relu(self.fc1(state.to(T.float32)))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent():
    def __init__(self, gamma, epsilon, alpha, input_dims, batch_size, actions, max_mem_size=100000, eps_min=.01, eps_dec = .995, opt=max):
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.alpha = alpha # learning rate
        self.input_dims = input_dims # list of input dimensions
        self.batch_size = batch_size # batch size
        self.mem_size = max_mem_size # maximum memory size
        self.eps_min = eps_min # minimum exploration rate
        self.eps_dec = eps_dec # exploration rate decay
        self.action_space = {a: i for i, a in enumerate(actions)} # dict of {action: index}
        self.mem_cntr = 0 # memory counter
        self.opt = opt # function to determine what is considered optimal, i.e., max or min

        self.Q_eval = DQNetwork(self.alpha, action_space=self.action_space, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=bool) # memory of states
        self._state_memory = np.zeros((self.mem_size, *input_dims), dtype=bool) # memory of next states
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32) # memory of actions
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) # memory of rewards
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool) # memory of terminal states

    def store_transition(self, state, action, reward, _state, done):
        index = self.mem_cntr % self.mem_size # index of memory to store transition
        self.state_memory[index] = state # store state
        self._state_memory[index] = _state # store next state
        self.reward_memory[index] = reward # store reward
        self.action_memory[index] = self.action_space[action] # store action
        self.terminal_memory[index] = done # store terminal state
        self.mem_cntr += 1 # increment memory counter

    def choose_action(self, observation, enabled_actions, force_greedy=False):
        if force_greedy or np.random.random() > self.epsilon:
            # Choose action greedily (exploit)
            state = T.tensor([observation]).to(self.Q_eval.device) # convert observation to tensor
            actions = self.Q_eval.forward(state) # get Q values for all actions
            action = self.opt(enabled_actions, key=lambda a: actions[0][self.action_space[a]]) # choose optimal enabled action
            q_value = actions[0][self.action_space[action]].item() # return Q value for chosen action
        else:
            # Choose action randomly (explore)
            action = np.random.choice(enabled_actions) # choose random action
            q_value = None # return no Q value for random action
        return action, q_value

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return # don't learn until memory is full

        self.Q_eval.optimizer.zero_grad() # zero gradients
        max_mem = min(self.mem_cntr, self.mem_size) # maximum memory to sample from
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # sample batch from memory
        batch_index = np.arange(self.batch_size, dtype=np.int32) # batch indices

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) # convert batch of states to tensor
        _state_batch = T.tensor(self._state_memory[batch]).to(self.Q_eval.device) # convert batch of next states to tensor
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device) # convert batch of rewards to tensor
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device) # convert batch of terminal states to tensor
        action_batch = self.action_memory[batch] # batch of actions

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] # get Q values for all actions
        q_next = self.Q_eval.forward(_state_batch) # get Q values for all actions in next state
        q_next[terminal_batch] = 0.0 # set next Q values of terminal states to 0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] # calculate target Q values

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device) # calculate loss
        loss.backward() # backpropagate loss
        self.Q_eval.optimizer.step() # update weights

        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min # decay exploration rate
