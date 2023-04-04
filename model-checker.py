import sys
from importlib import util
from timeit import default_timer as timer

from mdp import *


debug = False
debug_value_iteration = False
debug_value_iteration_initialization = False
debug_precomputation = False


class ModelChecker():
    def __init__(self, network: Network) -> None:
        self.network = network

        self.states = self.explore(network.get_initial_state())
        self.properties = network.properties

        if debug:
            print('Explored the following states:')
            for state in self.states:
                print(state)
            print()

            print('Model checks for the following properties:')
            for i in range(len(self.properties)):
                print(f'{i}: {self.properties[i]}')
            print()
    
    def explore(self, s: State, explored: List[State] = []) -> List[State]:
        if s not in explored:
            explored.append(s)
            for a in self.network.get_transitions(s):
                for delta in self.network.get_branches(s, a):
                    explored = self.explore(self.network.jump(s, a, delta), explored)
        
        return explored

    def precompute_Smin0(self, expression: int) -> List[State]:
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
                    R += [_s for _s in S if _s.state == s.state]
                    # R.append(s)
        
        if debug_precomputation:
            print(f'{network.properties[expression]} Smin0:')
            for s in S:
                if s not in R:
                    print(s)
            print()
        
        return [s for s in S if s not in R] # S \ R
    
    def precompute_Smin1(self, expression: int) -> List[State]:
        S = self.states
        R = [s for s in S if s not in self.precompute_Smin0(expression)]
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
                    R = list(filter(lambda _s: _s.state != s.state, R))
                    # R.remove(s)
        
        if debug_precomputation:
            print(f'{network.properties[expression]} Smin1:')
            for s in R:
                print(s)
            print()

        return R
    
    def precompute_Smax0(self, expression: int) -> List[State]:
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
                    R += [_s for _s in S if _s.state == s.state]
                    # R.append(s)
        
        if debug_precomputation:
            print(f'{network.properties[expression]} Smax0:')
            for s in S:
                if s not in R:
                    print(s)
            print()
        
        return [s for s in S if s not in R] # S \ R
    
    def precompute_Smax1(self, expression: int) -> List[State]:
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
                        R += [_s for _s in S if _s.state == s.state]
                        # R.append(s)
        
        if debug_precomputation:
            print(f'{network.properties[expression]} Smax1:')
            for s in R:
                print(s)
            print()

        return R
    
    def value_iteration(self, n: int, expression: int) -> float:
        S = self.states
        exp = self.properties[expression].exp
        op = exp.op
        args = exp.args
        is_prob = exp is not None and op.startswith('p_')
        is_reach = exp is not None and op == 'exists' and (args[0].op == 'eventually' and args[0].args[0].op == 'ap' or args[0].op == 'until' and args[0].args[0].op == 'ap' and args[0].args[1].op == 'ap')
        safe_exp = None
        goal_exp = None
        reward_exp = None
        
        if is_reach or is_prob:
            safe_exp = args[0].args[0].args[0] if args[0].op == 'until' else None
            goal_exp = args[0].args[1].args[0] if args[0].op == 'until' else args[0].args[0].args[0]

        is_reward = exp is not None and op.startswith('e_') and args[1].op == 'ap'

        if is_reward:
            goal_exp = args[1].args[0]
            reward_exp = exp.args[0]
        
        S1 = self.precompute_Smin1(goal_exp) if op.endswith('_max') else self.precompute_Smax1(goal_exp)
        G = [s for s in S if self.network.get_expression_value(s, goal_exp)]
        if is_prob:
            # Probability value iteration initialization
            _v = {s: int(self.network.get_expression_value(s, goal_exp)) for s in S}
        elif is_reward:
            # Expected reward value iteration initialization
            _v = {s: 0 if s in G + S1 else float('inf') for s in S}
        else:
            raise ValueError('Unknown operator: {}'.format(op))
        
        if debug_value_iteration_initialization:
            print('S1:')
            for s in S1:
                print(s)
            print()

            print('G:')
            for s in G:
                print(s)
            print()
        
        for i in range(n):
            v = _v.copy() # v_i-1
            _v = {} # v_i
            for s in v:
                if is_prob:
                    paths = [sum([delta.probability * v[network.jump(s, a, delta)] for delta in self.network.get_branches(s, a)]) for a in self.network.get_transitions(s)]
                    _v[s] = min(paths) if op.endswith('_min') else max(paths)
                elif is_reward:
                    if s in G:
                        _v[s] = 0 # we have already reached G, we need not make any further transitions
                    elif s not in S1:
                        _v[s] = float('inf') # reward is infinite
                    else:
                        paths = [sum([reward_exp * delta.probability * v[network.jump(s, a, delta, [reward_exp])] for delta in self.network.get_branches(s, a)]) for a in self.network.get_transitions(s)]
                        _v[s] = min(paths) if op.endswith('_min') else max(paths)
            
            if debug_value_iteration:
                print(f'VI step {i}:')
                for s, P in _v.items():
                    print(f'{s}: {P}')
                print()
        
        return _v[network.get_initial_state()]


if __name__ == "__main__":
    # Load the model
    if len(sys.argv) < 2:
        print("Error: No model specified.")
        quit()
    print("Loading model from \"{0}\"...".format(sys.argv[1]), end = "", flush = True)
    spec = util.spec_from_file_location("model", sys.argv[1])
    model = util.module_from_spec(spec)
    spec.loader.exec_module(model)
    network = model.Network() # create network instance
    print(" done.")

    start_time = timer()

    model_checker = ModelChecker(network)

    if debug_precomputation:
        print(f'{network.properties[0]} Smin0:')
        for model in model_checker.precompute_Smin0(0):
            print(model)
        print()

        print(f'{network.properties[0]} Smin1:')
        for model in model_checker.precompute_Smin1(0):
            print(model)
        print()

        print(f'{network.properties[0]} Smax0:')
        for model in model_checker.precompute_Smax0(0):
            print(model)
        print()

        print(f'{network.properties[0]} Smax1:')
        for model in model_checker.precompute_Smax1(0):
            print(model)
        print()
    
    for property in network.properties:
        print(f'{property} = {model_checker.value_iteration(1000, network.properties.index(property))}')

    end_time = timer()
    print("Done in {0:.2f} seconds.".format(end_time - start_time))
