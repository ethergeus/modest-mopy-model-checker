#!/bin/python3
import sys
from importlib import util
from timeit import default_timer as timer
import argparse
import random

try:
    # Try to import the mdp module from the current directory (a simple example MDP with all classes implemented)
    # This is useful for testing the model checker and autocompletion in IDEs
    from mdp import *
except ImportError:
    pass


class ModelChecker():
    PROGRESS_INTERVAL = 2 # seconds
    MAX_RELATIVE_ERROR = 1e-6; # maximum relative error for value iteration
    Q_LEARNING_EXPLORATION = 0.1 # epsilon for Q-learning
    Q_LEARNING_RATE = 0.1 # alpha for Q-learning
    Q_LEARNING_DISCOUNT = 0.9 # gamma for Q-learning
    Q_LEARNING_RUNS = 20000 # number of runs for Q-learning

    def __init__(self, arguments) -> None:
        # Load the model
        if len(arguments) < 2:
            print("Error: No model specified.")
            quit()
        
        parser = argparse.ArgumentParser(
            prog='model-checker.py',
            description='Model checker for MDPs.',
            epilog='by Andrey and Alex (Group 2)')

        parser.add_argument('model', type=str, help='path to the model file')
        parser.add_argument('-p', '--properties', type=str, nargs='+', default=[], help='list of properties to check (default: all)')
        parser.add_argument('--value-iteration', action='store_true', help='use value iteration to evaluate properties')
        parser.add_argument('--relative-error', type=float, default=self.MAX_RELATIVE_ERROR, help=f'maximum relative error for value iteration (default: {self.MAX_RELATIVE_ERROR})')
        parser.add_argument('-k', '--max-iterations', type=int, default=0, help=f'maximum number of iterations for value iteration, takes precedence over relative error')
        parser.add_argument('--q-learning', action='store_true', help='use Q-learning to evaluate properties')
        parser.add_argument('-e', '--epsilon', type=float, default=self.Q_LEARNING_EXPLORATION, help=f'epsilon (exploration probability) for Q-learning (default: {self.Q_LEARNING_EXPLORATION})')
        parser.add_argument('-a', '--alpha', type=float, default=self.Q_LEARNING_RATE, help=f'alpha (learning rate) for Q-learning (default: {self.Q_LEARNING_RATE})')
        parser.add_argument('-g', '--gamma', type=float, default=self.Q_LEARNING_DISCOUNT, help=f'gamma (discount factor) for Q-learning (default: {self.Q_LEARNING_DISCOUNT})')

        self.args = parser.parse_args()

        print(f"Loading model from \"{self.args.model}\"...", end = "", flush = True)
        spec = util.spec_from_file_location("model", self.args.model)
        model = util.module_from_spec(spec)
        spec.loader.exec_module(model)
        self.states = [] # list of all states
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
        epsilon = self.args.epsilon

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
        
        return max(Q[SI].values()) if op.endswith('_max_s') else min(Q[SI].values())
    
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
            else:
                print("Error: No algorithm specified.")
                quit()
        
        end_time = timer()

        print("Done in {0:.2f} seconds.".format(end_time - start_time))
    
    def explore(self, explored):
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
                    for delta in self.network.get_branches(s, a):
                        # for each target state s' in A(s)
                        _s = self.network.jump(s, a, delta)
                        if _s not in explored:
                            # if s' is not in explored
                            explored.append(_s)
                            found = True # new state found
        
        return sorted(explored, key=lambda s: s.__str__()) # sort states by string representation

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
    
    def opt(self, op: str, val, key=lambda x: x) -> float:
        if op.find('min') != -1:
            return min(val, key=key)
        elif op.find('max') != -1:
            return max(val, key=key)
        else:
            raise Exception('Unknown operator: ' + op)

if __name__ == "__main__":
    ModelChecker(sys.argv)
