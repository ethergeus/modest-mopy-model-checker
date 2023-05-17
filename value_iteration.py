import sys

import utils

def calculate(model_checker, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
    # Explore state space using breadth-first search
    if len(model_checker.states) == 0:
        print('Exploring the state space...', end = '', flush = True)
        model_checker.states = model_checker.explore([model_checker.network.get_initial_state()])
        print(f' found a total of {len(model_checker.states)} states.')
    
    S = model_checker.states # all states
    G = [s for s in S if model_checker.network.get_expression_value(s, goal_exp)] # goal states
    if is_prob:
        # Probability value iteration initialization
        _v = {s: int(model_checker.network.get_expression_value(s, goal_exp)) for s in S}
    elif is_reward:
        # Expected reward value iteration initialization
        # In case of a maximum reward we will pre-compute states where the minimum probability is 1
        # In case of a minimum reward we will pre-compute states where the maximum probability is 1
        S1 = utils.precompute_Smin1(model_checker.network, model_checker.states, goal_exp) if op.find('max') != -1 else utils.precompute_Smax1(model_checker.network, model_checker.states, goal_exp)

        # Initialize value iteration, 0 where if s is in S1, +inf otherwise
        _v = {s: 0 if s in S1 else float('inf') for s in S}

        # Sanity check, G should be a subset of S1
        for s in G:
            assert s in S1
    else:
        raise ValueError('Unknown operator: {}'.format(op))
    
    k = model_checker.args.max_iterations
    error = model_checker.args.relative_error
    
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
                    paths = [sum([delta.probability * v[model_checker.network.jump(s, a, delta)] for delta in model_checker.network.get_branches(s, a)]) for a in model_checker.network.get_transitions(s)]
                    _v[s] = utils.opt(op, paths)
            elif is_reward:
                if s in G:
                    _v[s] = 0 # we have already reached G, we need not make any further transitions
                elif s not in S1:
                    _v[s] = float('inf') # reward is infinite
                else:
                    paths = []
                    for a in model_checker.network.get_transitions(s):
                        r = 0
                        for delta in model_checker.network.get_branches(s, a):
                            reward = [reward_exp]
                            r += delta.probability * (v[model_checker.network.jump(s, a, delta, reward)] + reward[0])
                        paths.append(r)
                    _v[s] = utils.opt(op, paths)
        
        if k == 0:
            if all(_v[s] == float('inf') or _v[s] == 0 or abs(_v[s] - v[s]) / _v[s] < error for s in v):
                break
    
    print(' done. ', end = '', flush = True)

    return _v[model_checker.network.get_initial_state()]