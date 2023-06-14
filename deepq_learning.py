import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import utils

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_enabled_actions', 'goal_state')) # transition tuple

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
                yield 0
    
    def _obs(self, state):
        variables = self.network.variables
        
        for var in range(len(variables)):
            if variables[var].name in self.ignore:
                continue
            if variables[var].minValue is not None and variables[var].maxValue is not None and (self.onehot_all or variables[var].name in self.onehot):
                yield from self._onehot(state, variables, var)
            else:
                yield state.get_variable_value(var)
        
        locations = list(self._loc(state))
        for i, component in enumerate(self.network.components):
            for location in range(len(component.transition_counts)):
                yield locations[i] == location
    
    def _onehot(self, state, variables, var):
        for i in range(variables[var].minValue, variables[var].maxValue + 1):
            yield state.get_variable_value(var) == i
    
    def __len__(self):
        return len(self.data)

class Action():
    def __init__(self, model_checker, transitions=None):
        self.network = model_checker.network
        self.data = [False] * sum([sum(component.transition_counts) for component in self.network.components]) if transitions is None else list(self._act(transitions))
    
    def _act(self, transitions):
        for i, component in enumerate(self.network.components):
            k = max(component.transition_counts)
            for j in range(k):
                yield transitions[i] == j
    
    def __eq__(self, other):
        return self.data == other.data
    
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
    def __init__(self, device, maxlen=100000):
        self.device = device
        self.buffer = deque([], maxlen=maxlen)
    
    def push(self, state, action, reward, next_state, next_enabled_actions, goal_state):
        state = T.tensor([state.data], dtype=T.float32, device=self.device)
        action = T.tensor([action.data], dtype=T.float32, device=self.device)
        reward = T.tensor([reward], dtype=T.float32, device=self.device)
        next_state = T.tensor([next_state.data], dtype=T.float32, device=self.device)
        next_enabled_actions = T.tensor([action.data for action in next_enabled_actions], dtype=T.float32, device=self.device)
        goal_state = T.tensor([goal_state], dtype=T.bool, device=self.device)
        self.buffer.append(Transition(state, action, reward, next_state, next_enabled_actions, goal_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, input_dims, fc_dims):
        super(DQNetwork, self).__init__()

        # Action space, used to map actions to indices of output layer
        self.input_dims = input_dims # list of input dimensions
        self.fc_dims = fc_dims # list of fully connected layer dimensions

        # Define layers
        self.fc = nn.ModuleList([nn.Linear(*self.input_dims, self.fc_dims[0])]) # first fully connected layer
        for i in range(1, len(self.fc_dims)):
            # Add fully connected layers
            self.fc.append(nn.Linear(self.fc_dims[i-1], self.fc_dims[i]))
        self.fc.append(nn.Linear(self.fc_dims[-1], 1)) # output layer
    
    def forward(self, x):
        # Forward pass through network
        for layer in self.fc[:-1]:
            x = F.relu(layer(x)) # pass through fully connected layers
        return self.fc[-1](x) # return output of last layer (Q value)


class DQAgent():
    def __init__(self, gamma, epsilon, alpha,
                 input_dims, fc_dims=[512, 512, 512],
                 max_mem_size=100000, batch_size=64,
                 eps_min=.01, eps_dec=.995,
                 opt=max,
                 verbose=False,
                 plot=False):
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.alpha = alpha # learning rate
        self.input_dims = input_dims # list of input dimensions
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

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # device to use for training

        if self.verbose:
            print(f'Creating policy and target networks with input dimensions {self.input_dims} and fully connected layer dimensions {self.fc_dims}')
        
        self.policy_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims).to(self.device) # policy network
        self.target_net = DQNetwork(input_dims=input_dims, fc_dims=fc_dims).to(self.device) # target network
        self.target_net.load_state_dict(self.policy_net.state_dict()) # copy policy network weights to target network
        self.parameters = self.policy_net.parameters() # list of parameters for all layers
        self.optimizer = optim.AdamW(self.parameters, lr=alpha, amsgrad=True) # AdamW optimizer
        self.buffer = ReplayBuffer(self.device, max_mem_size) # replay buffer
        self.criterion = nn.MSELoss() # Mean squared error loss

    def select_action(self, obs, enabled_actions, greedy=False):
        if greedy or random.uniform(0, 1) > self.epsilon:
            state = T.tensor([obs.data] * len(enabled_actions), dtype=T.float32, device=self.device)
            actions = T.tensor([action.data for action in enabled_actions], dtype=T.float32, device=self.device)
            with T.no_grad():
                q_values = self.policy_net(T.cat((state, actions), dim=1)) # concatenate state-action pairs
            opt = self.torch_argopt(q_values) # get index of optimal action
            return enabled_actions[opt], q_values[opt].item() # return action and Q-value
        else:
            return random.choice(enabled_actions), None
        
    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return 0
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)
        goal_state_mask = T.cat(batch.goal_state)
        
        with T.no_grad():
            next_state_values = T.stack([T.tensor(0, dtype=T.float32, device=self.device) if len(actions) == 0 else self.torch_opt(self.target_net(T.cat((state.repeat(len(actions), 1), actions), dim=1))) for state, actions in zip(batch.next_state, batch.next_enabled_actions)])
        
        next_state_values[goal_state_mask] = 0 # Q value for goal state is zero

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch # expected Q-values
        
        self.optimizer.zero_grad()
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
    
    input_dims = [len(Observation(model_checker, SI)) + len(Action(model_checker))]
    
    # Deep Q-learning agent
    agent = DQAgent(gamma=model_checker.args.gamma,
                    epsilon=model_checker.args.epsilon_start,
                    alpha=model_checker.args.alpha,
                    input_dims=input_dims,
                    fc_dims=model_checker.args.fc_dims,
                    max_mem_size=model_checker.args.max_mem_size,
                    batch_size=model_checker.args.batch_size,
                    eps_min=model_checker.args.epsilon_min,
                    eps_dec=model_checker.args.epsilon_decay,
                    opt=utils._opt(op),
                    verbose=model_checker.args.verbose,
                    plot=model_checker.args.plot)
    
    if model_checker.args.verbose:
        print(f'One-hot variables: {"all" if model_checker.args.onehot_all else model_checker.args.onehot}')
    
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
            enabled_actions = [Action(model_checker, a.transitions) for a in A] # enabled actions
            action, _ = agent.select_action(obs, enabled_actions) # choose action from network
            a = next(a for a in A if Action(model_checker, a.transitions) == action) # get action
            D = model_checker.network.get_branches(s, a) # possible transitions
            delta = random.choices(D, weights=[delta.probability for delta in D])[0] # choose transition randomly
            reward = [reward_exp]
            _s = model_checker.network.jump(s, a, delta, reward) # r, s' = sample(s, a)
            _A = model_checker.network.get_transitions(_s)
            next_enabled_actions = [Action(model_checker, a.transitions) for a in _A]
            
            goal_state = model_checker.network.get_expression_value(_s, goal_exp) # we reached a goal state (Q := 0)
            self_loop = _s == s and len(A) == 1 and len(D) == 1 # the only possible transition is to itself, i.e., s' = s (tau loop)
            deadlock = len(_A) == 0 # the number of outgoing transitions from s' = 0 (deadlock)
            
            _obs = Observation(model_checker, _s) # get observation from state
            
            agent.buffer.push(obs, action, reward[0], _obs, next_enabled_actions, goal_state) # store transition in replay buffer
            
            loss = agent.optimize_model() # train agent
            
            # Soft update of the target network's weights
            agent.soft_update(model_checker.args.tau)
        
        _, q_value = agent.select_action(Observation(model_checker, SI), [Action(model_checker, a.transitions) for a in model_checker.network.get_transitions(SI)], greedy=True)
    
    if model_checker.args.verbose:
        print(f'Finished: Q = {q_value:.2f}, loss = {loss:.2f}, epsilon = {agent.epsilon:.2f}, run = {run}{" " * 16}')
    if model_checker.args.plot:
        results.push(q_value, loss)
        results.plot()
    return q_value