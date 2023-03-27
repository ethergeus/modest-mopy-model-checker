# Run as python checker-demo.py model.py
# Requires Python 3.7 or newer

import sys
from importlib import util
from timeit import default_timer as timer

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

# Print some information about the model and the initial state
start_time = timer()
print("* The model has", str(len(network.properties)), "properties.")
print("* The first property is:", str(network.properties[0]))
is_reach = network.properties[0].exp is not None and network.properties[0].exp.op == "exists" and (network.properties[0].exp.args[0].op == "eventually" and network.properties[0].exp.args[0].args[0].op == "ap" or network.properties[0].exp.args[0].op == "until" and network.properties[0].exp.args[0].args[0].op == "ap" and network.properties[0].exp.args[0].args[1].op == "ap")
safe_exp = -1
goal_exp = -1
if is_reach:
	safe_exp = network.properties[0].exp.args[0].args[0].args[0] if network.properties[0].exp.args[0].op == "until" else -1
	goal_exp = network.properties[0].exp.args[0].args[1].args[0] if network.properties[0].exp.args[0].op == "until" else network.properties[0].exp.args[0].args[0].args[0]
print("* The first property is", "" if is_reach else " not", " a reachability property.", sep = "")
reward_exps = []
is_reward = network.properties[0].exp is not None and network.properties[0].exp.op == "e_min_s" and network.properties[0].exp.args[1].op == "ap"
if is_reward:
	reward_exps = [network.properties[0].exp.args[0]]
print("* The first property is", "" if is_reward else " not", " a cost-optimal reachability property.", sep = "")
initial_state = network.get_initial_state()
print("* The initial state is:", str(initial_state))
if safe_exp != -1:
	print("* The initial state is ", "" if network.get_expression_value(initial_state, safe_exp) else "not ", "a safe state.", sep = "")
if goal_exp != -1:
	print("* The initial state is ", "" if network.get_expression_value(initial_state, goal_exp) else "not ", "a goal state.", sep = "")

initial_state_transitions = network.get_transitions(initial_state)
print("* The initial state has", str(len(initial_state_transitions)), "transitions.")
print("* The first transition out of the initial state is labelled:", network.transition_labels[initial_state_transitions[0].label])
next_state = network.jump_np(initial_state, initial_state_transitions[0], reward_exps)
if len(reward_exps) != 0:
	print("* The transition has a reward of", reward_exps[0], "for the first property.")
print("* The transition leads to state:", str(next_state))

print()
print("* The second transition out of the initial state is labelled:", network.transition_labels[initial_state_transitions[1].label])
next_state = network.jump_np(initial_state, initial_state_transitions[1], reward_exps)
if len(reward_exps) != 0:
	print("* The transition has a reward of", reward_exps[0], "for the first property.")
print("* The transition leads to state:", str(next_state))

print()
next_state_transitions = network.get_transitions(next_state)
print("* The first transition out of the 'second' state is labelled:", network.transition_labels[next_state_transitions[0].label])
next_next_state = network.jump_np(next_state, next_state_transitions[0], reward_exps)
if len(reward_exps) != 0:
	print("* The transition has a reward of", reward_exps[0], "for the first property.")
print("* The transition leads to state:", str(next_next_state))

print()
print("* The second transition out of the 'second' state is labelled:", network.transition_labels[next_state_transitions[1].label])
next_next_state = network.jump(initial_state, next_state_transitions[1], network.get_branches(next_state, next_state_transitions[1])[1], reward_exps)
if len(reward_exps) != 0:
	print("* The transition has a reward of", reward_exps[0], "for the first property.")
print("* The transition leads to state:", str(next_next_state))

print()
end_time = timer()
print("Done in {0:.2f} seconds.".format(end_time - start_time))
