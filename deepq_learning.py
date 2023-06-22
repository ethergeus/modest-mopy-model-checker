import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
import numpy as np

import utils

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_enabled_actions', 'goal_state', 'self_loop', 'deadlock')) # transition tuple

class Observation():
    def __init__(self, model_checker, state):
        self.network = model_checker.network # MDP model network
        self.onehot_all = model_checker.args.onehot_all # encode all bounded variables one-hot
        self.onehot = model_checker.args.onehot # encode certain variables one-hot
        self.ignore = model_checker.args.ignore # ignore certain variables in encoding
        self.data = list(self._obs(state))
    
    def _loc(self, state):
        for component in self.network.components:
            try:
                # Get the location for a given automaton
                yield getattr(state, type(component).__name__.replace('Automaton', '_location'))
            except AttributeError:
                yield 0 # automaton has been optimized to not have a location property -- return 0 as the only location
    
    def _obs(self, state):
        variables = self.network.variables # all variables in the entire network
        
        for var in range(len(variables)):
            if variables[var].name in self.ignore:
                continue # the currently selected variable is in the ignore list
            if variables[var].minValue is not None and variables[var].maxValue is not None and (self.onehot_all or variables[var].name in self.onehot):
                yield from self._onehot(state, variables, var) # construct a series of onehot-encoded neurons
            else:
                yield state.get_variable_value(var) # return the ordinal value of the variable
        
        locations = list(self._loc(state))
        for i, component in enumerate(self.network.components):
            for location in range(len(component.transition_counts)):
                yield locations[i] == location
    
    def _onehot(self, state, variables, var):
        # Encode a variable as one-hot, i.e., a variable with range 0..4 and value 1 is encoded as [False, True, False, False]
        for i in range(variables[var].minValue, variables[var].maxValue + 1):
            yield state.get_variable_value(var) == i
    
    def __eq__(self, other):
        return self.data == other.data
    
    def __len__(self):
        return len(self.data)

class Action():
    def __init__(self, model_checker, transition, enabled_transitions, output_dim):
        self.network = model_checker.network
        self.n = output_dim
        self.t = transition.transitions
        self.data = self._modulo_act(transition, enabled_transitions) if model_checker.args.table else list(self._act(transition))
    
    def _act(self, transition):
        for i, component in enumerate(self.network.components):
            for label in self.network.transition_labels.keys():
                yield transition.label == label
            t = max(component.transition_counts)
            for j in range(t):
                yield transition.transitions[i] == j
    
    def _modulo_act(self, transition, enabled_transitions):
        t = enabled_transitions.index(transition)
        k = len(enabled_transitions)
        return [random.choice(range(t, self.n, k))]
    
    def __eq__(self, other):
        return self.t == other.t
    
    def __len__(self):
        return len(self.data)

class Results(object):
    def __init__(self, maxlen=100000):
        self.q_value = deque([], maxlen=maxlen)
        self.loss = deque([], maxlen=maxlen)
    
    def push(self, q_values, loss):
        self.q_value.append(q_values)
        self.loss.append(loss)
    
    def plot(self):
        plt.figure()
        plt.title(f'Q-values over time (answer: Q = {np.mean(np.array(self.q_value)[-11:-1])})')
        plt.xlabel('Iterations')
        plt.ylabel('Q-value')
        plt.plot(self.q_value, marker='.', linestyle='')
        plt.yscale('log')
        plt.savefig(os.path.join('plot', 'q_values.pdf'))
        plt.show()
        
        plt.figure()
        plt.title('Loss over time')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(self.loss, marker='.', linestyle='')
        plt.savefig(os.path.join('plot', 'loss.pdf'))
        plt.show()

class ReplayBuffer(object):
    def __init__(self, device, maxlen=100000):
        self.device = device
        self.buffer = deque([], maxlen=maxlen)
    
    def push(self, state, action, reward, next_state, next_enabled_actions, goal_state, self_loop, deadlock):
        # Construct tensors from raw data and save to replay buffer for later sampling
        state = T.tensor([state.data], dtype=T.float32, device=self.device)
        action = T.tensor([action.data], dtype=T.int64, device=self.device)
        reward = T.tensor([reward], dtype=T.float32, device=self.device)
        next_state = T.tensor([next_state.data], dtype=T.float32, device=self.device)
        next_enabled_actions = T.tensor([action.data for action in next_enabled_actions], dtype=T.float32, device=self.device)
        goal_state = T.tensor([goal_state], dtype=T.bool, device=self.device)
        self_loop = T.tensor([self_loop], dtype=T.bool, device=self.device)
        deadlock = T.tensor([deadlock], dtype=T.bool, device=self.device)
        self.buffer.append(Transition(state, action, reward, next_state, next_enabled_actions, goal_state, self_loop, deadlock))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, input_dims, fc_dims, output_dim=1):
        super(DQNetwork, self).__init__()

        self.input_dims = input_dims # list of input dimensions
        self.fc_dims = fc_dims # list of fully connected layer dimensions
        self.output_dim = output_dim # dimension of output layer

        # Define layers
        input_layer = nn.Linear(*self.input_dims, self.fc_dims[0])
        self.fc = nn.ModuleList([input_layer]) # first fully connected layer
        for i in range(1, len(self.fc_dims)):
            # Add fully connected layers between input and output layers
            layer = nn.Linear(self.fc_dims[i-1], self.fc_dims[i])
            self.fc.append(layer)
        output_layer = nn.Linear(self.fc_dims[-1], self.output_dim)
        self.fc.append(output_layer) # output layer
    
    def forward(self, x):
        # Forward pass through network
        for layer in self.fc[:-1]:
            x = F.relu(layer(x)) # pass through fully connected layers
        return self.fc[-1](x) # return output of last layer (Q value)


class DQAgent():
    def __init__(self, gamma, epsilon, alpha,
                 input_dims, fc_dims=[512, 512, 512], output_dim=1,
                 max_mem_size=100000, batch_size=64,
                 eps_min=.01, eps_dec=.995,
                 double_q=False,
                 table=False,
                 opt=max,
                 verbose=False,
                 plot=False):
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.alpha = alpha # learning rate
        self.input_dims = input_dims # list of input dimensions
        self.fc_dims = fc_dims # list of fully connected layer dimensions
        self.output_dim = output_dim # output layer dimension
        self.mem_size = max_mem_size # maximum memory size
        self.batch_size = batch_size # batch size
        self.eps_min = eps_min # minimum exploration rate
        self.eps_dec = eps_dec # exploration rate decay
        self.double_q = double_q # whether to apply double deep Q learning
        self.table = table # whether to output a Q-table as output layer
        self.opt = opt # function to determine what is considered optimal, i.e., max or min
        self.torch_opt = T.max if opt == max else T.min # torch equivalent of opt
        self.torch_argopt = T.argmax if opt == max else T.argmin # torch equivalent of argopt
        self.verbose = verbose # whether to print debug information
        self.plot = plot # whether to plot results

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device to use for training

        if self.verbose:
            print(f'Creating policy and target networks with input dimensions {self.input_dims}, fully connected layer dimensions {self.fc_dims} and output layer dimension {output_dim}')
        
        self.policy_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims, output_dim=output_dim).to(self.device) # policy network
        self.target_net = self.policy_net if not double_q else DQNetwork(input_dims=input_dims, fc_dims=fc_dims, output_dim=output_dim).to(self.device) # target network
        if double_q:
            self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy network weights to target network
        self.parameters = self.policy_net.parameters() # list of parameters for all layers
        self.optimizer = optim.Adam(self.parameters, lr=alpha) # Adam optimizer
        self.buffer = ReplayBuffer(self.device, max_mem_size) # replay buffer
        self.criterion = nn.MSELoss() # Mean squared error loss (L2 loss function)

    def select_action(self, obs, enabled_actions, greedy=False):
        if greedy or random.uniform(0, 1) > self.epsilon:
            with T.no_grad():
                if self.table:
                    state = T.tensor([obs.data], dtype=T.float32, device=self.device)
                    q_values = self.policy_net(state).squeeze(0)
                else:
                    actions = T.tensor([action.data for action in enabled_actions], dtype=T.float32, device=self.device)
                    state = T.tensor([obs.data] * len(enabled_actions), dtype=T.float32, device=self.device)
                    q_values = self.policy_net(T.cat((state, actions), dim=1)) # concatenate state-action pairs
            opt = self.torch_argopt(q_values) # get index of optimal action
            return enabled_actions[opt % len(enabled_actions)], q_values[opt % len(enabled_actions)].item() # return action and Q-value
        else:
            return random.choice(enabled_actions), None
        
    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return 0
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        next_state_batch = T.cat(batch.next_state)
        reward_batch = T.cat(batch.reward)
        goal_state_mask = T.cat(batch.goal_state)
        self_loop_mask = T.cat(batch.self_loop)
        deadlock_mask = T.cat(batch.deadlock)
        
        next_state_values = T.zeros(self.batch_size, device=self.device)
        mask = T.logical_and(~goal_state_mask, ~deadlock_mask)
        
        with T.no_grad():
            if self.table:
                next_state_values = self.torch_opt(self.target_net(next_state_batch), 1)[0] # contains the optimal Q values for the non-final states
                next_state_values[goal_state_mask] = 0 # Q value for goal state is zero
            else:
                for i, (condition, state, actions) in enumerate(zip(mask, batch.next_state, batch.next_enabled_actions)):
                    if condition:
                        next_state_values[i] = self.torch_opt(self.target_net(T.cat((state.repeat(len(actions), 1), actions), dim=1)))

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # expected Q-values
        
        self.optimizer.zero_grad()
        
        if self.table:
            state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        else:
            state_action_values = self.policy_net(T.cat((state_batch, action_batch), dim=1)).squeeze(1) # current Q-values

        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        
        # Optimize the model
        loss.backward()
        
        self.optimizer.step()
        
        self.epsilon_decay(self.eps_dec)
        
        return loss.item()
    
    def epsilon_decay(self, decay):
        self.epsilon = max(self.epsilon * decay, self.eps_min) # decay exploration rate
    
    def soft_update(self, tau):
        # Soft update target network parameters
        # θ′ ← τ θ + (1 − τ)θ′
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

def learn(model_checker, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
    if not is_reward:
        return None # Q-learning only works for expected reward properties
    
    if model_checker.args.plot:
        results = Results()
    
    SI = model_checker.network.get_initial_state() # initial state
    A0 = model_checker.network.get_transitions(SI) # initial transitions
    
    counts = {i: component.transition_counts for i, component in enumerate(model_checker.network.components)}
    output_dim = np.sum([max(k) for k in counts.values()]) if model_checker.args.table else 1
    input_dims = [len(Observation(model_checker, SI))] if model_checker.args.table else [len(Observation(model_checker, SI)) + len(Action(model_checker, random.choice(A0), A0, output_dim))]
    
    # Deep Q-learning agent
    agent = DQAgent(gamma=model_checker.args.gamma,
                    epsilon=model_checker.args.epsilon_start,
                    alpha=model_checker.args.alpha,
                    input_dims=input_dims,
                    fc_dims=model_checker.args.fc_dims,
                    output_dim=output_dim,
                    max_mem_size=model_checker.args.max_mem_size,
                    batch_size=model_checker.args.batch_size,
                    eps_min=model_checker.args.epsilon_min,
                    eps_dec=model_checker.args.epsilon_decay,
                    double_q=model_checker.args.double_q,
                    table=model_checker.args.table,
                    opt=utils._opt(op),
                    verbose=model_checker.args.verbose,
                    plot=model_checker.args.plot)
    
    if model_checker.args.verbose:
        print(f'One-hot variables: {"all bounded" if model_checker.args.onehot_all else model_checker.args.onehot}')
    
    q_value, loss = 0, 0 # Q value and loss
    k = model_checker.args.max_iterations # maximum number of iterations
    t0 = timer()
    for run in range(k if k != 0 else model_checker.Q_LEARNING_RUNS):
        _s = SI # reset state to initial state
        _obs = Observation(model_checker, _s) # observation of initial state

        if model_checker.args.plot:
            results.push(q_value, loss)
        if model_checker.args.verbose:
            t1 = timer()
            if t1 - t0 > model_checker.PROGRESS_INTERVAL:
                print(f'Progress: Q = {q_value:.2f}, loss = {loss:.2f}, epsilon = {agent.epsilon:.2f}, run = {run}{" " * 16}', end='\r', flush=True)
                t0 = t1
        
        goal_state, self_loop, deadlock = False, False, False
        while not goal_state and not self_loop and not deadlock:
            s, obs = _s, _obs # update state and observation
            A = model_checker.network.get_transitions(s) # possible actions
            enabled_actions = [Action(model_checker, a, A, output_dim) for a in A] # enabled actions
            action, _ = agent.select_action(obs, enabled_actions, output_dim) # choose action from network
            a = next(a for a in A if Action(model_checker, a, A, output_dim) == action) # get action
            D = model_checker.network.get_branches(s, a) # possible transitions
            delta = random.choices(D, weights=[delta.probability for delta in D])[0] # choose transition randomly
            reward = [reward_exp]
            _s = model_checker.network.jump(s, a, delta, reward) # r, s' = sample(s, a)
            _A = model_checker.network.get_transitions(_s)
            next_enabled_actions = [Action(model_checker, a, _A, output_dim) for a in _A]
            
            goal_state = model_checker.network.get_expression_value(_s, goal_exp) # we reached a goal state (Q := 0)
            self_loop = _s == s and len(A) == 1 and len(D) == 1 # the only possible transition is to itself, i.e., s' = s (tau loop)
            deadlock = len(_A) == 0 # the number of outgoing transitions from s' = 0 (deadlock)
            
            _obs = Observation(model_checker, _s) # get observation from state
            
            agent.buffer.push(obs, action, reward[0], _obs, next_enabled_actions, goal_state, self_loop, deadlock) # store transition in replay buffer
            
            loss = agent.optimize_model() # train agent
            
            # Soft update of the target network's weights
            if model_checker.args.double_q:
                agent.soft_update(model_checker.args.tau)
        
        _, q_value = agent.select_action(Observation(model_checker, SI), [Action(model_checker, a, A0, output_dim) for a in A0], greedy=True)
    
    if model_checker.args.verbose:
        print(f'Finished: Q = {q_value:.2f}, loss = {loss:.2f}, epsilon = {agent.epsilon:.2f}, run = {run}{" " * 16}')
    if model_checker.args.plot:
        results.push(q_value, loss)
        results.plot()
    return q_value