import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import utils

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'enabled_actions')) # transition tuple

class Results(object):
    def __init__(self, maxlen=100000) -> None:
        self.q_value = deque([], maxlen=maxlen)
        self.loss = deque([], maxlen=maxlen)
    
    def push(self, q_values, loss):
        self.q_value.append(q_values)
        self.loss.append(loss)
    
    def plot(self):
        plt.figure()
        plt.title('Q-values over time')
        plt.xlabel('Iterations')
        plt.ylabel('Q-value')
        plt.plot(self.q_value, marker='.', linestyle='')
        plt.show()
        plt.figure()
        plt.title('Loss over time')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(self.loss, marker='.', linestyle='')
        plt.show()

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
        enabled_actions = T.tensor(enabled_actions, dtype=T.float32)
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
        self.activation = nn.Sigmoid() # activation function for fully connected layers

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
            x = self.activation(layer(x)) # pass through fully connected layers
        return self.fc[-1](x) # return output of last layer (Q values for all actions)


class DQAgent():
    PUNISHMENT = -1e6 # punishment for invalid actions

    def __init__(self, gamma, epsilon, alpha,
                 input_dims, num_actions, fc_dims=[256, 256],
                 max_mem_size=100000, batch_size=64,
                 eps_min=.01, eps_dec = .995,
                 opt=max,
                 verbose=False,
                 plot=False,
                 punish_invalid=False,
                 ignore_invalid=True):
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
        self.plot = plot # whether to plot results
        self.punish_invalid = punish_invalid # whether to punish agent for invalid actions
        self.ignore_invalid = ignore_invalid # whether to ignore invalid actions

        if not punish_invalid and not ignore_invalid:
            print('WARNING: Agent is neither punishing nor ignoring invalid actions. This may lead to unexpected behavior.')

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device to use for training

        if self.verbose:
            print(f'Creating policy and target networks with input dimensions {self.input_dims}, fully connected layer dimensions {self.fc_dims}, and {self.num_actions} actions as output layer')
        self.policy_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims, num_actions=num_actions).to(self.device) # policy network
        self.target_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims, num_actions=num_actions).to(self.device) # target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy network weights to target network
        self.parameters = nn.ModuleList(self.policy_net.fc).parameters() # list of parameters for all layers
        self.optimizer = optim.SGD(self.parameters, lr=alpha) # Adam optimizer
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
        disabled_actions_mask = T.tensor(tuple(map(lambda a: tuple([i not in a for i in range(self.num_actions)]), batch.enabled_actions)), dtype=T.bool).to(self.device) # mask of disabled actions (n x num_actions)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch) # get Q values for all actions taken in batch (n x 1)
        with T.no_grad():
            target_eval = self.target_net(next_state_batch) # get Q values for all actions in next state (n x num_actions)
            # Filter out disabled actions
            if self.ignore_invalid:
                # Make disabled actions have unattractive Q values
                target_eval[disabled_actions_mask] = -np.inf if self.opt == max else np.inf # set disabled actions to -inf if max opt, inf if min opt
            next_state_values = self.torch_opt(target_eval, 1)[0] # get optimal Q values of all actions in next state (n x 1)
        if self.punish_invalid:
            pass # TODO: implement punishment for invalid actions
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # calculate expected state action values
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1)) # calculate loss
        self.optimizer.zero_grad() # zero gradients
        loss.backward() # backpropagate loss
        self.optimizer.step() # update parameters
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min # decay exploration rate

        return loss.item() # return loss
    
    def soft_update(self, tau):
        # Soft update target network parameters
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

def learn(model_checker, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
    if not is_reward:
        return None # Q-learning only works for expected reward properties
    
    if model_checker.args.plot:
        results = Results()
    
    SI = model_checker.network.get_initial_state() # initial state
    
    obs = list(utils.state2obs(model_checker.network, SI, onehot_all_vars=model_checker.args.onehot_all, onehot_vars=model_checker.args.onehot))
    input_dims = [len(obs)]
    components = model_checker.network.components # automaton components
    num_actions = max(*components, key=lambda c: c.transition_counts[0]).transition_counts[0] if len(components) > 1 else components[0].transition_counts[0] # number of actions is the maximum number of transitions in any component
    
    # Deep Q-learning agent
    agent = DQAgent(gamma=model_checker.args.gamma,
                    epsilon=model_checker.args.epsilon_start,
                    alpha=model_checker.args.alpha,
                    input_dims=input_dims,
                    num_actions=num_actions,
                    fc_dims=model_checker.args.fc_dims,
                    max_mem_size=model_checker.args.max_mem_size,
                    batch_size=model_checker.args.batch_size,
                    eps_min=model_checker.args.epsilon_min,
                    eps_dec=model_checker.args.epsilon_decay,
                    opt=utils._opt(op),
                    verbose=model_checker.args.verbose,
                    plot=model_checker.args.plot,
                    punish_invalid=model_checker.args.punish_invalid,
                    ignore_invalid=model_checker.args.ignore_invalid)
    
    if model_checker.args.verbose:
        print(f'One-hot variables: {True if model_checker.args.onehot_all else model_checker.args.onehot}')
    
    q_value, loss = 0, 0 # Q value and loss
    k = model_checker.args.max_iterations # maximum number of iterations
    t0 = timer()
    for run in range(k if k != 0 else model_checker.Q_LEARNING_RUNS):
        _s = SI # reset state to initial state
        _obs = list(utils.state2obs(model_checker.network, _s, onehot_all_vars=model_checker.args.onehot_all, onehot_vars=model_checker.args.onehot)) # observation of initial state

        if model_checker.args.plot:
            results.push(q_value, loss)
        if model_checker.args.verbose:
            t1 = timer()
            if t1 - t0 > model_checker.PROGRESS_INTERVAL:
                print(f'Progress: Q = {q_value:.2f}, loss = {loss:.2f}, epsilon = {agent.epsilon:.2f}, run = {run}', end = '\r', flush = True)
                t0 = t1
        
        while not model_checker.network.get_expression_value(_s, goal_exp):
            s, obs = _s, _obs # update state and observation
            A = model_checker.network.get_transitions(s) # possible actions
            enabled_actions = [a.transitions[0] for a in A] # enabled actions
            action, _ = agent.choose_action(obs, enabled_actions) # choose action from network
            a = next(a for a in A if a.transitions[0] == action) # get action from index
            assert a in A # sanity check, chosen action should be in action space
            D = model_checker.network.get_branches(s, a) # possible transitions
            delta = random.choices(D, weights=[delta.probability for delta in D])[0] # choose transition randomly
            reward = [reward_exp]
            _s = model_checker.network.jump(s, a, delta, reward) # r, s' = sample(s, a)

            # If we have reached a terminal state, break
            # The only possible transition is to itself, i.e., s' = s (tau loop)
            if _s == s and len(A) == 1 and len(D) == 1:
                break # if term(s')

            _obs = list(utils.state2obs(model_checker.network, _s, onehot_all_vars=model_checker.args.onehot_all, onehot_vars=model_checker.args.onehot)) # get observation from state

            agent.buffer.push_mdp_tensor(obs, action, reward[0], _obs, enabled_actions) # store transition in replay buffer

            loss = agent.learn() # train agent

            # Soft update of the target network's weights
            agent.soft_update(model_checker.args.tau)
    
        enabled_actions = [a.transitions[0] for a in A] # enabled actions in initial state
        _, q_value = agent.choose_action(list(utils.state2obs(model_checker.network, SI, onehot_all_vars=model_checker.args.onehot_all, onehot_vars=model_checker.args.onehot)), enabled_actions, force_greedy=True)
    
    if model_checker.args.verbose:
        print(f'Finished: Q = {q_value:.2f}, loss = {loss:.2f}, epsilon = {agent.epsilon:.2f}, run = {run}')
    if model_checker.args.plot:
        results.push(q_value, loss)
        results.plot()
    return q_value