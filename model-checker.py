#!/usr/bin/env python3
import sys
from importlib import util
from timeit import default_timer as timer
import argparse
import random

from deepq import DQAgent as Agent


class ModelChecker():
    PROGRESS_INTERVAL = 2 # seconds
    MAX_RELATIVE_ERROR = 1e-6; # maximum relative error for value iteration
    EPSILON_START = 1.0 # starting epsilon for deep Q-learning
    EPSILON_MIN = 0.01 # ending epsilon for deep Q-learning
    EPSILON_DECAY = 0.9999 # epsilon decay for deep Q-learning
    Q_LEARNING_EXPLORATION = 0.1 # epsilon for Q-learning
    Q_LEARNING_RATE = 0.1 # alpha for Q-learning
    Q_LEARNING_DISCOUNT = 1 # gamma for Q-learning
    Q_LEARNING_RUNS = 5000 # number of runs for Q-learning
    MAX_MEM_SIZE = 100000 # maximum memory size for deep Q-learning
    BATCH_SIZE = 64 # batch size for deep Q-learning
    UPDATE_INTERVAL = 1000 # number of steps between target network updates for deep Q-learning
    TAU = 0.005 # tau for soft target network updates

    def __init__(self, arguments) -> None:
        # Load the model
        if len(arguments) < 2:
            print("Error: No model specified.")
            quit()
        
        parser = argparse.ArgumentParser(
            prog='model-checker.py',
            description='Model checker for MDPs.',
            epilog='Originally created by Andrey and Alex (Group 2) for the course Probabilistic Model Checking, continued by Andrey Antonowycz for Capita Selecta')

        parser.add_argument('model', type=str, help='path to the model file')
        parser.add_argument('-p', '--properties', type=str, nargs='+', default=[], help='list of properties to check (default: all)')
        parser.add_argument('--value-iteration', action='store_true', help='use value iteration to evaluate properties')
        parser.add_argument('--relative-error', type=float, default=self.MAX_RELATIVE_ERROR, help=f'maximum relative error for value iteration (default: {self.MAX_RELATIVE_ERROR})')
        parser.add_argument('-k', '--max-iterations', type=int, default=0, help=f'maximum number of iterations for value iteration, takes precedence over relative error')
        parser.add_argument('--q-learning', action='store_true', help='use Q-learning to evaluate properties')
        parser.add_argument('--epsilon-start', type=float, default=self.EPSILON_START, help=f'initial epsilon (exploration probability) (default: {self.EPSILON_START})')
        parser.add_argument('--epsilon-min', type=float, default=self.EPSILON_MIN, help=f'minimum epsilon (exploration probability) (default: {self.EPSILON_MIN})')
        parser.add_argument('--epsilon-decay', type=float, default=self.EPSILON_DECAY, help=f'epsilon decay rate (default: {self.EPSILON_DECAY})')
        parser.add_argument('-a', '--alpha', type=float, default=self.Q_LEARNING_RATE, help=f'alpha (learning rate) for Q-learning (default: {self.Q_LEARNING_RATE})')
        parser.add_argument('-g', '--gamma', type=float, default=self.Q_LEARNING_DISCOUNT, help=f'gamma (discount factor) for Q-learning (default: {self.Q_LEARNING_DISCOUNT})')

        # Deep Q-learning parameters
        parser.add_argument('--deep-q-learning', action='store_true', help='use deep Q-learning to evaluate properties')
        parser.add_argument('--max-mem-size', type=int, default=self.MAX_MEM_SIZE, help=f'maximum size of the replay memory (default: {self.MAX_MEM_SIZE})')
        parser.add_argument('--batch-size', type=int, default=self.BATCH_SIZE, help=f'batch size for training (default: {self.BATCH_SIZE})')
        parser.add_argument('--update-interval', type=int, default=self.UPDATE_INTERVAL, help=f'number of steps between target network updates (default: {self.UPDATE_INTERVAL})')
        parser.add_argument('--tau', type=float, default=self.TAU, help=f'tau for soft target network updates (default: {self.TAU})')

        parser.add_argument('--verbose', '-v', action='store_true', help='print progress information when available')

        self.args = parser.parse_args()

        print(f"Loading model from \"{self.args.model}\"...", end = "", flush = True)
        spec = util.spec_from_file_location("model", self.args.model)
        model = util.module_from_spec(spec)
        spec.loader.exec_module(model)
        self.states = [] # list of all states
        self.transitions = [] # list of all transitions
        self.network = model.Network() # create network instance
        self.properties = self.network.properties
        print(" done.")

        # Perform model checking on the specified properties
        self.check_properties(self.args.properties)
    
    def _value_iteration(self, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
        # Explore state space using breadth-first search
        if len(self.states) == 0:
            print('Exploring the state space...', end = '', flush = True)
            self.states = self.explore([self.network.get_initial_state()])
            print(f' found a total of {len(self.states)} states.')
        
        S = self.states # all states
        G = [s for s in S if self.network.get_expression_value(s, goal_exp)] # goal states
        if is_prob:
            # Probability value iteration initialization
            _v = {s: int(self.network.get_expression_value(s, goal_exp)) for s in S}
        elif is_reward:
            # Expected reward value iteration initialization
            # In case of a maximum reward we will pre-compute states where the minimum probability is 1
            # In case of a minimum reward we will pre-compute states where the maximum probability is 1
            S1 = self.precompute_Smin1(goal_exp) if op.find('max') != -1 else self.precompute_Smax1(goal_exp)

            # Initialize value iteration, 0 where if s is in S1, +inf otherwise
            _v = {s: 0 if s in S1 else float('inf') for s in S}

            # Sanity check, G should be a subset of S1
            for s in G:
                assert s in S1
        else:
            raise ValueError('Unknown operator: {}'.format(op))
        
        k = self.args.max_iterations
        error = self.args.relative_error
        
        # Value iteration
        print('Performing value iteration...', end = '', flush = True)
        for _ in range(k if k != 0 else sys.maxsize):
            v = _v # v_i-1
            _v = {} # v_i
            for s in v:
                if is_prob:
                    if is_reach and s in G:
                        _v[s] = v[s] # tau
                    else:
                        paths = [sum([delta.probability * v[self.network.jump(s, a, delta)] for delta in self.network.get_branches(s, a)]) for a in self.network.get_transitions(s)]
                        _v[s] = self.opt(op, paths)
                elif is_reward:
                    if s in G:
                        _v[s] = 0 # we have already reached G, we need not make any further transitions
                    elif s not in S1:
                        _v[s] = float('inf') # reward is infinite
                    else:
                        paths = []
                        for a in self.network.get_transitions(s):
                            r = 0
                            for delta in self.network.get_branches(s, a):
                                reward = [reward_exp]
                                r += delta.probability * (v[self.network.jump(s, a, delta, reward)] + reward[0])
                            paths.append(r)
                        _v[s] = self.opt(op, paths)
            
            if k == 0:
                if all(_v[s] == float('inf') or _v[s] == 0 or abs(_v[s] - v[s]) / _v[s] < error for s in v):
                    break
        
        print(' done. ', end = '', flush = True)

        return _v[self.network.get_initial_state()]
    
    def _q_learning(self, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
        if not is_reward:
            return None # Q-learning only works for expected reward properties

        SI = self.network.get_initial_state() # initial state
        
        Q = {SI: {a.label: 0 for a in self.network.get_transitions(SI)}} # Q(s, a) initialized to 0 for all transitions in initial state SI

        k = self.args.max_iterations
        alpha = self.args.alpha
        gamma = self.args.gamma
        epsilon = self.args.epsilon_start

        for _ in range(k if k != 0 else self.Q_LEARNING_RUNS):
            _s = SI # reset state to initial state

            # While not in a goal state
            while not self.network.get_expression_value(_s, goal_exp):
                s = _s
                A = self.network.get_transitions(s)
                if random.uniform(0, 1) < epsilon:
                    a = random.choice(A) # Exploration
                else:
                    a = self.opt(op, A, key=lambda a: Q[s][a.label]) # Exploitation
                
                # Take random transition
                D = self.network.get_branches(s, a)
                delta = random.choices(D, weights=[delta.probability for delta in D])[0]
                
                # Take transition and extract reward
                reward = [reward_exp]
                _s = self.network.jump(s, a, delta, reward) # r, s' = sample(s, a)

                # Check if we already have a Q value for s'
                if _s not in Q:
                    Q[_s] = {a.label: 0 for a in self.network.get_transitions(_s)}

                # Update Q value in table
                Q[s][a.label] += alpha * (reward[0] + gamma * self.opt(op, Q[_s].values()) - Q[s][a.label])

                # If we have reached a terminal state, break
                # The only possible transition is to itself, i.e., s' = s (tau loop)
                if _s == s and len(A) == 1 and len(D) == 1:
                    break # if term(s')
            
            # Decay epsilon
            epsilon = max(self.args.epsilon_min, epsilon * self.args.epsilon_decay)
        
        return self.opt_fn(op)(Q[SI].values())
    
    def _deep_q_learning(self, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
        if not is_reward:
            return None # Q-learning only works for expected reward properties
        
        SI = self.network.get_initial_state() # initial state
        
        # Explore state space using breadth-first search
        if len(self.transitions) == 0 or len(self.states) == 0:
            print('Exploring the state space...', end = '', flush = True)
            self.states, self.transitions = self.explore([self.network.get_initial_state()])
            print(f' found a total of {len(self.states)} states and {len(self.transitions)} transitions.')
        
        fc_dims = [len(self.states), 128, 128]
        label2index = {a: i for i, a in enumerate(self.transitions)}
        index2label = {i: a for i, a in enumerate(self.transitions)}
        agent = Agent(gamma=self.args.gamma, epsilon=self.args.epsilon_start, alpha=self.args.alpha, input_dims=[len(self.states)], num_actions=len(label2index), fc_dims=fc_dims,
                      max_mem_size=self.args.max_mem_size, batch_size=self.args.batch_size, eps_min=self.args.epsilon_min, eps_dec=self.args.epsilon_decay, opt=self.opt_fn(op), verbose=self.args.verbose)
        q_value = 0
        k = self.args.max_iterations
        t0 = timer()
        for run in range(k if k != 0 else self.Q_LEARNING_RUNS):
            done = False # whether we have reached a terminal state
            s = SI # reset state to initial state
            obs = [self.states.index(s) == i for i in range(len(self.states))]

            if self.args.verbose:
                t1 = timer()
                if t1 - t0 > self.PROGRESS_INTERVAL:
                    print(f'Progress: Q = {q_value:.2f}, epsilon = {agent.epsilon:.2f}, run = {run}', end = '\r', flush = True)
                    t0 = t1
            
            while not done:
                A = self.network.get_transitions(s) # possible actions
                enabled_actions = [label2index[a.label] for a in A] # enabled actions
                action, _ = agent.choose_action(obs, enabled_actions) # choose action label from network
                a = next(a for a in A if a.label == index2label[action]) # get action from label
                assert a in A # sanity check, chosen action should be in action space
                D = self.network.get_branches(s, a) # possible transitions
                delta = random.choices(D, weights=[delta.probability for delta in D])[0] # choose transition randomly
                reward = [reward_exp]
                _s = self.network.jump(s, a, delta, reward) # r, s' = sample(s, a)
                _obs = [self.states.index(_s) == i for i in range(len(self.states))] # get observation from state

                # If we have reached a terminal state, break
                # The only possible transition is to itself, i.e., s' = s (tau loop) or the goal expression is satisfied
                if _s == s and len(A) == 1 and len(D) == 1 or self.network.get_expression_value(_s, goal_exp):
                    done = True # if term(s')
                
                agent.buffer.push_mdp_tensor(obs, action, reward[0], _obs, done) # store transition in replay buffer

                s, obs = _s, _obs # update state and observation
                agent.learn() # train agent

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                policy_net_state_dict = agent.policy_net.state_dict()
                target_net_state_dict = agent.target_net.state_dict()
                for key in policy_net_state_dict.keys():
                    target_net_state_dict[key] = self.args.tau * policy_net_state_dict[key] + (1 - self.args.tau) * target_net_state_dict[key]
                agent.target_net.load_state_dict(target_net_state_dict)
        
            _, q_value = agent.choose_action([self.states.index(SI) == i for i in range(len(self.states))], [label2index[a.label] for a in self.network.get_transitions(SI)], force_greedy=True)
        
        return q_value
    
    def check_properties(self, properties = []) -> None:
        if len(properties) == 0:
            # No properties specified, check all properties
            properties = self.properties
        else:
            properties = [property for property in self.properties if property.name in properties]

        start_time = timer()

        # Parse properties
        for property in properties:
            exp = property.exp # expression to evaluate
            op = exp.op # operator of the expression
            args = exp.args # arguments of the expression

            # Is the expression a probability expression?
            is_prob = exp is not None and op.startswith('p_')

            # Is the expression a reachability expression?
            is_reach = exp is not None and op == 'exists' and (args[0].op == 'eventually' and args[0].args[0].op == 'ap' or args[0].op == 'until' and args[0].args[0].op == 'ap' and args[0].args[1].op == 'ap')
            
            # Is the expression a reward expression?
            is_reward = exp is not None and op.startswith('e_') and args[1].op == 'ap'

            safe_exp = None # expression for the safe states (before until)
            goal_exp = None # expression for the goal states
            reward_exp = None # expression for the reward
            
            if is_reach or is_prob:
                # Extract useful expressions from the reachability or probability expression
                safe_exp = args[0].args[0].args[0] if args[0].op == 'until' else None
                goal_exp = args[0].args[1].args[0] if args[0].op == 'until' else args[0].args[0].args[0]

            if is_reward:
                # Extract useful expressions from the reward expression
                goal_exp = args[1].args[0]
                reward_exp = exp.args[0]
            
            # Perform the actual computation
            if self.args.value_iteration:
                print(f'{property} = {self._value_iteration(op, is_prob, is_reach, is_reward, goal_exp, reward_exp)}')
            elif self.args.q_learning:
                print(f'{property} = {self._q_learning(op, is_prob, is_reach, is_reward, goal_exp, reward_exp)}')
            elif self.args.deep_q_learning:
                print(f'{property} = {self._deep_q_learning(op, is_prob, is_reach, is_reward, goal_exp, reward_exp)}')
            else:
                print("Error: No algorithm specified.")
                quit()
        
        end_time = timer()

        print("Done in {0:.2f} seconds.".format(end_time - start_time))
    
    def explore(self, explored):
        labels = [] # list of transition labels
        found = True # flag to indicate if new states were found
        t0 = timer() # timer to print progress
        while found:
            t1 = timer()
            if t1 - t0 > self.PROGRESS_INTERVAL:
                # print progress every PROGRESS_INTERVAL seconds
                print(f' {int(len(explored) / 1000)}k..', end = '', flush = True)
                t0 = t1
            found = False # reset flag
            for s in explored:
                # for each state s in explored
                for a in self.network.get_transitions(s):
                    # for each action a in A(s)
                    if a.label not in labels:
                        labels.append(a.label)
                    for delta in self.network.get_branches(s, a):
                        # for each target state s' in A(s)
                        _s = self.network.jump(s, a, delta)
                        if _s not in explored:
                            # if s' is not in explored
                            explored.append(_s)
                            found = True # new state found
        
        return sorted(explored, key=lambda s: s.__str__()), sorted(labels) # sort states by string representation and labels alphabetically

    def precompute_Smin0(self, expression: int):
        print('Pre-computing Smin0... ', end = '', flush = True)
        S = self.states
        R = [s for s in S if self.network.get_expression_value(s, expression)]
        _R = [] # R' from the paper

        while set(R) != set(_R):
            _R = R.copy()
            for s in S:
                # for each state s in S
                forall_a = True
                for a in self.network.get_transitions(s):
                    # for each action a in A(s)
                    exists_delta = False
                    for delta in self.network.get_branches(s, a):
                        # for every target state s' in R'
                        _s = self.network.jump(s, a, delta) # s' from the paper
                        if _s in _R and delta.probability > 0:
                            # there exists s' in R' such that a -> s' with probability > 0
                            exists_delta = True
                            break
                    
                    if not exists_delta:
                        forall_a = False
                        break

                if forall_a and s not in R:
                    R.append(s)
        
        return sorted([s for s in S if s not in R], key=lambda s: s.__str__()) # S \ R
    
    def precompute_Smin1(self, expression: int):
        print('Pre-computing Smin1... ', end = '', flush = True)
        S = self.states
        Smin0 = self.precompute_Smin0(expression)
        R = [s for s in S if s not in Smin0]
        _R = [] # R' from the paper

        while set(R) != set(_R):
            _R = R.copy()
            for s in R:
                # for each state s in S
                exists_a = False
                for a in self.network.get_transitions(s):
                    # for each action a in A(s)
                    exists_delta = False
                    for delta in self.network.get_branches(s, a):
                        # for every target state s' in R'
                        _s = self.network.jump(s, a, delta)
                        if _s not in _R and delta.probability > 0:
                            # there exists s' not in R' such that a -> s' with probability > 0
                            exists_delta = True
                            break
                    
                    if exists_delta:
                        exists_a = True
                        break
                
                if exists_a and s in R:
                    R.remove(s)

        return sorted(R, key=lambda s: s.__str__())
    
    def precompute_Smax0(self, expression: int):
        print('Pre-computing Smax0... ', end = '', flush = True)
        S = self.states
        R = [s for s in S if self.network.get_expression_value(s, expression)]
        _R = [] # R' from the paper
        
        while set(R) != set(_R):
            _R = R.copy()
            for s in S:
                # for each state s in S
                exists_a = False
                for a in self.network.get_transitions(s):
                    # for each action a in A(s)
                    exists_delta = False
                    for delta in self.network.get_branches(s, a):
                        # for every target state s' in R'
                        _s = self.network.jump(s, a, delta)
                        if _s in _R and delta.probability > 0:
                            # there exists s' in R' such that a -> s' with probability > 0
                            exists_delta = True
                            break
                    
                    if exists_delta:
                        exists_a = True
                        break
                
                if exists_a and s not in R:
                    R.append(s)
        
        return sorted([s for s in S if s not in R], key=lambda s: s.__str__()) # S \ R
    
    def precompute_Smax1(self, expression: int):
        print('Pre-computing Smax1... ', end = '', flush = True)
        S = self.states
        T = [s for s in S if self.network.get_expression_value(s, expression)]
        R = S.copy()
        _R = [] # R' from the paper
        __R = [] # R'' from the paper

        while set(R) != set(_R):
            _R = R.copy()
            R = T.copy()
            while set(R) != set(__R):
                __R = R.copy()
                for s in S:
                    # for each state s in S
                    exists_a = False
                    for a in self.network.get_transitions(s):
                        # for each action a in A(s)
                        forall_s = True
                        exists_s = False

                        for delta in self.network.get_branches(s, a):
                            # for every target state s' in R'
                            _s = self.network.jump(s, a, delta)
                            if _s not in _R or delta.probability == 0:
                                # there does not exist s' in R' such that a -> s' with probability > 0
                                forall_s = False # reject transition
                                break
                            if _s in __R and delta.probability > 0:
                                # there exists s' in R'' such that a -> s' with probability > 0
                                exists_s = True

                        if forall_s and exists_s:
                            exists_a = True
                            break
                    
                    if exists_a and s not in R:
                        R.append(s)
        
        return sorted(R, key=lambda s: s.__str__())
    
    def opt_fn(self, op: str):
        if op.find('min') != -1:
            return min
        elif op.find('max') != -1:
            return max
        else:
            raise Exception('Unknown operator: ' + op)
    
    def opt(self, op: str, val, key=lambda x: x) -> float:
        return self.opt_fn(op)(val, key=key)

if __name__ == "__main__":
    ModelChecker(sys.argv)
