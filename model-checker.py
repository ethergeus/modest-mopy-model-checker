import sys
from importlib import util
from timeit import default_timer as timer

from mdp import *


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
            
            print(f'VI step {i}:')
            for state, prob in states_new.items():
                print(f'{state}: {prob}')
            print()
        
        probability = states_new[network.get_initial_state()]
        print(f'{self.properties[expression]}: {probability}')
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
    print(model_checker.value_iteration(5, 0))
    print(model_checker.value_iteration(5, 1))

    end_time = timer()
    print("Done in {0:.2f} seconds.".format(end_time - start_time))
