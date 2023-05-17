import random

import utils

def _q_learning(model_checker, op: str, is_prob: bool, is_reach: bool, is_reward: bool, goal_exp, reward_exp) -> float:
    if not is_reward:
        return None # Q-learning only works for expected reward properties

    SI = model_checker.network.get_initial_state() # initial state
    
    Q = {SI: {a.label: 0 for a in model_checker.network.get_transitions(SI)}} # Q(s, a) initialized to 0 for all transitions in initial state SI

    k = model_checker.args.max_iterations
    alpha = model_checker.args.alpha
    gamma = model_checker.args.gamma
    epsilon = model_checker.args.epsilon_start

    for _ in range(k if k != 0 else model_checker.Q_LEARNING_RUNS):
        _s = SI # reset state to initial state

        # While not in a goal state
        while not model_checker.network.get_expression_value(_s, goal_exp):
            s = _s
            A = model_checker.network.get_transitions(s)
            if random.uniform(0, 1) < epsilon:
                a = random.choice(A) # Exploration
            else:
                a = utils.opt(op, A, key=lambda a: Q[s][a.label]) # Exploitation
            
            # Take random transition
            D = model_checker.network.get_branches(s, a)
            delta = random.choices(D, weights=[delta.probability for delta in D])[0]
            
            # Take transition and extract reward
            reward = [reward_exp]
            _s = model_checker.network.jump(s, a, delta, reward) # r, s' = sample(s, a)

            # Check if we already have a Q value for s'
            if _s not in Q:
                Q[_s] = {a.label: 0 for a in model_checker.network.get_transitions(_s)}

            # Update Q value in table
            Q[s][a.label] += alpha * (reward[0] + gamma * utils.opt(op, Q[_s].values()) - Q[s][a.label])

            # If we have reached a terminal state, break
            # The only possible transition is to itself, i.e., s' = s (tau loop)
            if _s == s and len(A) == 1 and len(D) == 1:
                break # if term(s')
        
        # Decay epsilon
        epsilon = max(model_checker.args.epsilon_min, epsilon * model_checker.args.epsilon_decay)
    
    return utils._opt(op)(Q[SI].values())