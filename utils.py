import itertools

def _opt(op: str):
    if op.find('min') != -1:
        return min
    elif op.find('max') != -1:
        return max
    else:
        raise Exception('Unknown operator: ' + op)

def opt(op: str, val, key=lambda x: x) -> float:
    return _opt(op)(val, key=key)

def state2obs(network, state, onehot_all_vars=False, onehot_vars=[]):
    return list(itertools.chain(
        *[onehot(state, var) if onehot_all_vars or network.variables[var].name in onehot_vars
            else [state.get_variable_value(var)] for var in range(len(network.variables))]))

def onehot(network, state, var):
    v = network.variables[var]
    return [state.get_variable_value(var) == i for i in range(v.minValue, v.maxValue + 1)]

def precompute_Smin0(network, states, expression: int):
    print('Pre-computing Smin0... ', end = '', flush = True)
    S = states
    R = [s for s in S if network.get_expression_value(s, expression)]
    _R = [] # R' from the paper

    while set(R) != set(_R):
        _R = R.copy()
        for s in S:
            # for each state s in S
            forall_a = True
            for a in network.get_transitions(s):
                # for each action a in A(s)
                exists_delta = False
                for delta in network.get_branches(s, a):
                    # for every target state s' in R'
                    _s = network.jump(s, a, delta) # s' from the paper
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

def precompute_Smin1(network, states, expression: int):
    print('Pre-computing Smin1... ', end = '', flush = True)
    S = states
    Smin0 = precompute_Smin0(network, states, expression)
    R = [s for s in S if s not in Smin0]
    _R = [] # R' from the paper

    while set(R) != set(_R):
        _R = R.copy()
        for s in R:
            # for each state s in S
            exists_a = False
            for a in network.get_transitions(s):
                # for each action a in A(s)
                exists_delta = False
                for delta in network.get_branches(s, a):
                    # for every target state s' in R'
                    _s = network.jump(s, a, delta)
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

def precompute_Smax0(network, states, expression: int):
    print('Pre-computing Smax0... ', end = '', flush = True)
    S = states
    R = [s for s in S if network.get_expression_value(s, expression)]
    _R = [] # R' from the paper
    
    while set(R) != set(_R):
        _R = R.copy()
        for s in S:
            # for each state s in S
            exists_a = False
            for a in network.get_transitions(s):
                # for each action a in A(s)
                exists_delta = False
                for delta in network.get_branches(s, a):
                    # for every target state s' in R'
                    _s = network.jump(s, a, delta)
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

def precompute_Smax1(network, states, expression: int):
    print('Pre-computing Smax1... ', end = '', flush = True)
    S = states
    T = [s for s in S if network.get_expression_value(s, expression)]
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
                for a in network.get_transitions(s):
                    # for each action a in A(s)
                    forall_s = True
                    exists_s = False

                    for delta in network.get_branches(s, a):
                        # for every target state s' in R'
                        _s = network.jump(s, a, delta)
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