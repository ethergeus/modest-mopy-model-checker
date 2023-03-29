import sys
from importlib import util
from timeit import default_timer as timer

from mdp import *


debug = False


class ModelChecker():
    def __init__(self, network: Network) -> None:
        self.network = network
        self.automaton = network._aut_Main

        self.states = self.explore(network.get_initial_state(), [])
        print('Explored the following states:')
        for state in self.states:
            print(state)
        print()

        self.properties = network.properties
        print('Checking for the following properties:')
        for property in self.properties:
            print(property)
        print()
    
    def explore(self, state: State, memory: List[State]) -> List[State]:
        if state not in memory:
            memory.append(state)
            for transition in self.network.get_transitions(state):
                for branch in self.network.get_branches(state, transition):
                    memory = self.explore(self.network.jump(state, transition, branch), memory)
        
        return memory

    def precompute_Smin0(self, expression: int) -> List[State]:
        S = self.states
        R = [state for state in S if self.network.get_expression_value(state, expression)]
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
        
        return [state for state in S if state not in R] # S \ R
    
    def precompute_Smin1(self, expression: int) -> List[State]:
        S = self.states
        R = [state for state in S if state not in self.precompute_Smin0(expression)]
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

        return R
    
    def precompute_Smax0(self, expression: int) -> List[State]:
        S = self.states
        R = [state for state in S if self.network.get_expression_value(state, expression)]
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
        
        return [state for state in S if state not in R] # S \ R
    
    def precompute_Smax1(self, expression: int) -> List[State]:
        S = self.states
        T = [state for state in S if self.network.get_expression_value(state, expression)]
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
        return R
    
    def value_iteration(self, n: int, expression: int) -> float:
        states_new = {state: 1 if self.network.get_expression_value(state, expression) else 0 for state in self.states}
        print('Initializing value iteration with initial array:')
        for state, prob in states_new.items():
            print(f'{state}: {prob}')
        print()
        
        for i in range(n):
            states_old = states_new
            states_new = {}
            op = self.properties[expression].exp.op
            for state in states_old:
                probabilities = []
                for transition in self.network.get_transitions(state):
                    prob = 0
                    for branch in self.network.get_branches(state, transition):
                        prob += branch.probability * states_old[network.jump(state, transition, branch)]
                    probabilities.append(prob)
                if op == 'p_min':
                    val = min(probabilities)
                elif op == 'p_max':
                    val = max(probabilities)
                else:
                    raise KeyError
                states_new[state] = val
            
            if debug:
                print(f'VI step {i}:')
                for state, prob in states_new.items():
                    print(f'{state}: {prob}')
                print()
        
        probability = states_new[network.get_initial_state()]
        print(f'{self.properties[expression]} = {probability}')
        print()
        return probability


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
    print(model_checker.value_iteration(1000, 0))
    print('Smin0:')
    for model in model_checker.precompute_Smin0(0):
        print(model)
    print()

    print('Smin1:')
    for model in model_checker.precompute_Smin1(0):
        print(model)
    print()

    print('Smax0:')
    for model in model_checker.precompute_Smax0(0):
        print(model)
    print()

    print('Smax1:')
    for model in model_checker.precompute_Smax1(0):
        print(model)
    print()

    end_time = timer()
    print("Done in {0:.2f} seconds.".format(end_time - start_time))
