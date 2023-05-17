#!/usr/bin/env python3
import sys
from importlib import util
from timeit import default_timer as timer
import argparse


class ModelChecker():
    PROGRESS_INTERVAL = 2 # seconds
    MAX_RELATIVE_ERROR = 1e-6; # maximum relative error for value iteration
    EPSILON_START = 1.0 # starting epsilon for deep Q-learning
    EPSILON_MIN = 0.01 # ending epsilon for deep Q-learning
    EPSILON_DECAY = 0.995 # epsilon decay for deep Q-learning
    Q_LEARNING_EXPLORATION = 0.1 # epsilon for Q-learning
    Q_LEARNING_RATE = 0.1 # alpha for Q-learning
    Q_LEARNING_DISCOUNT = 1 # gamma for Q-learning
    Q_LEARNING_RUNS = 5000 # number of runs for Q-learning
    MAX_MEM_SIZE = 100000 # maximum memory size for deep Q-learning
    BATCH_SIZE = 64 # batch size for deep Q-learning
    UPDATE_INTERVAL = 1000 # number of steps between target network updates for deep Q-learning
    TAU = 0.005 # tau for soft target network updates
    FC_DIMS = [256, 256] # fully connected layer dimensions for deep Q-learning
    ONEHOT_ALL = False # whether to use one-hot encoding for states
    ONEHOT = ['state'] # variables to use one-hot encoding for

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
        parser.add_argument('--fc-dims', type=int, nargs='+', default=self.FC_DIMS, help=f'dimensions of the fully connected layers (default: {self.FC_DIMS})')
        parser.add_argument('--onehot-all', action='store_true', help='use one-hot encoding for all variables')
        parser.add_argument('--onehot', type=str, nargs='+', default=self.ONEHOT, help=f'variables to use one-hot encoding for (default: {self.ONEHOT})')

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
                import value_iteration
                print(f'{property} = {value_iteration._value_iteration(self, op, is_prob, is_reach, is_reward, goal_exp, reward_exp)}')
            elif self.args.q_learning:
                import q_learning
                print(f'{property} = {q_learning._q_learning(self, op, is_prob, is_reach, is_reward, goal_exp, reward_exp)}')
            elif self.args.deep_q_learning:
                import deepq_learning
                print(f'{property} = {deepq_learning._deep_q_learning(self, op, is_prob, is_reach, is_reward, goal_exp, reward_exp)}')
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
        
        return sorted(explored, key=lambda s: s.__str__()) # sort states by string representation


if __name__ == "__main__":
    ModelChecker(sys.argv)
