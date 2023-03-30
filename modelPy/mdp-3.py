# mdp-3

from __future__ import annotations
from typing import List, Union, Optional
import math

class ExplorationError(Exception):
	__slots__ = ("message")
	
	def __init__(self, message: str):
		self.message = message

class VariableInfo(object):
	__slots__ = ("name", "component", "type", "minValue", "maxValue")
	
	def __init__(self, name: str, component: Optional[int], type: str, minValue = None, maxValue = None):
		self.name = name
		self.component = component
		self.type = type
		self.minValue = minValue
		self.maxValue = maxValue

# States
class State(object):
	__slots__ = ("successful", "failure", "Choose_location")
	
	def get_variable_value(self, variable: int):
		if variable == 0:
			return self.successful
		elif variable == 1:
			return self.failure
		elif variable == 2:
			return self.Choose_location
	
	def copy_to(self, other: State):
		other.successful = self.successful
		other.failure = self.failure
		other.Choose_location = self.Choose_location
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.successful == other.successful and self.failure == other.failure and self.Choose_location == other.Choose_location
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.successful)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.failure)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.Choose_location)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "successful = " + str(self.successful)
		result += ", failure = " + str(self.failure)
		result += ", Choose_location = " + str(self.Choose_location)
		result += ")"
		return result

# Transients
class Transient(object):
	__slots__ = ("reward")
	
	def copy_to(self, other: Transient):
		other.reward = self.reward
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.reward == other.reward
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.reward)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "reward = " + str(self.reward)
		result += ")"
		return result

# Automaton: Choose
class ChooseAutomaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [2, 2, 1, 1, 1]
		self.transition_labels = [[1, 2], [1, 2], [0], [0], [0]]
		self.branch_counts = [[1, 1], [1, 2], [1], [1], [2]]
	
	def set_initial_values(self, state: State) -> None:
		state.Choose_location = 0
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = state.Choose_location
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[state.Choose_location]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[state.Choose_location][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = state.Choose_location
		if location == 2 or location == 3 or location == 4:
			return True
		elif location == 0:
			return True
		elif location == 1:
			return True
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = state.Choose_location
		if location == 2 or location == 3 or location == 4:
			return None
		elif location == 0:
			return None
		elif location == 1:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[state.Choose_location][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = state.Choose_location
		if location == 0:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return 1
			elif transition == 1:
				if True:
					return (1 / 2)
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return 1
			else:
				raise IndexError
		elif location == 3:
			if transition == 0:
				return 1
			else:
				raise IndexError
		elif location == 4:
			if transition == 0:
				if branch == 0:
					return (1 / 10)
				elif branch == 1:
					return (9 / 10)
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == 0:
			location = state.Choose_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.Choose_location = 4
				elif transition == 1:
					if branch == 0:
						target_state.Choose_location = 1
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_state.Choose_location = 0
				elif transition == 1:
					if branch == 0:
						target_state.successful = True
						target_transient.reward = 8
						target_state.Choose_location = 3
					elif branch == 1:
						target_state.failure = True
						target_state.Choose_location = 2
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_state.Choose_location = 2
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_state.Choose_location = 3
			elif location == 4:
				if transition == 0:
					if branch == 0:
						target_state.successful = True
						target_transient.reward = 2
						target_state.Choose_location = 3
					elif branch == 1:
						target_state.Choose_location = 0

class PropertyExpression(object):
	__slots__ = ("op", "args")
	
	def __init__(self, op: str, args: List[Union[int, float, PropertyExpression]]):
		self.op = op
		self.args = args
	
	def __str__(self):
		result = self.op + "("
		needComma = False
		for arg in self.args:
			if needComma:
				result += ", "
			else:
				needComma = True
			result += str(arg)
		return result + ")"

class Property(object):
	__slots__ = ("name", "exp")
	
	def __init__(self, name: str, exp: PropertyExpression):
		self.name = name
		self.exp = exp
	
	def __str__(self):
		return self.name + ": " + str(self.exp)

class Transition(object):
	__slots__ = ("sync_vector", "label", "transitions")
	
	def __init__(self, sync_vector: int, label: int = 0, transitions: List[int] = [-1]):
		self.sync_vector = sync_vector
		self.label = label
		self.transitions = transitions

class Branch(object):
	__slots__ = ("probability", "branches")
	
	def __init__(self, probability = 0.0, branches = [0]):
		self.probability = probability
		self.branches = branches

class Network(object):
	__slots__ = ("network", "model_type", "components", "transition_labels", "sync_vectors", "properties", "variables", "_initial_transient", "_aut_Choose")
	
	def __init__(self):
		self.network = self
		self.model_type = "mdp"
		self.transition_labels = { 0: "τ", 1: "s", 2: "r" }
		self.sync_vectors = [[0, 0], [1, 1], [2, 2]]
		self.properties = [
			Property("P1", PropertyExpression("p_max", [PropertyExpression("eventually", [PropertyExpression("ap", [0])])])),
			Property("P2", PropertyExpression("p_max", [PropertyExpression("eventually", [PropertyExpression("ap", [1])])])),
			Property("P3", PropertyExpression("p_min", [PropertyExpression("eventually", [PropertyExpression("ap", [0])])])),
			Property("P4", PropertyExpression("p_min", [PropertyExpression("eventually", [PropertyExpression("ap", [1])])])),
			Property("R1", PropertyExpression("e_max_s", [2, PropertyExpression("ap", [3])])),
			Property("R2", PropertyExpression("e_min_s", [2, PropertyExpression("ap", [0])]))
		]
		self.variables = [
			VariableInfo("successful", None, "bool"),
			VariableInfo("failure", None, "bool"),
			VariableInfo("Choose_location", 0, "int", 0, 4)
		]
		self._aut_Choose = ChooseAutomaton(self)
		self.components = [self._aut_Choose]
		self._initial_transient = self._get_initial_transient()
	
	def get_initial_state(self) -> State:
		state = State()
		state.successful = False
		state.failure = False
		self._aut_Choose.set_initial_values(state)
		return state
	
	def _get_initial_transient(self) -> Transient:
		transient = Transient()
		transient.reward = 0
		self._aut_Choose.set_initial_transient_values(transient)
		return transient
	
	def get_expression_value(self, state: State, expression: int):
		if expression == 0:
			return state.successful
		elif expression == 1:
			return state.failure
		elif expression == 2:
			return self.network._get_transient_value(state, "reward")
		elif expression == 3:
			return (state.successful or state.failure)
		else:
			raise IndexError
	
	def _get_jump_expression_value(self, state: State, transient: Transient, expression: int):
		if expression == 0:
			return state.successful
		elif expression == 1:
			return state.failure
		elif expression == 2:
			return transient.reward
		elif expression == 3:
			return (state.successful or state.failure)
		else:
			raise IndexError
	
	def _get_transient_value(self, state: State, transient_variable: str):
		# Query the automata for the current value of the transient variable
		result = self._aut_Choose.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		# No automaton has a value: return the transient variable's (cached) initial value
		return getattr(self._initial_transient, transient_variable)
	
	def get_transitions(self, state: State) -> List[Transition]:
		# Collect all automaton transitions, gathered by label
		transitions = []
		trans_Choose = [[], [], []]
		transition_count = self._aut_Choose.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_Choose.get_guard_value(state, i):
				trans_Choose[self._aut_Choose.get_transition_label(state, i)].append(i)
		# Match automaton transitions onto synchronisation vectors
		for svi in range(len(self.sync_vectors)):
			sv = self.sync_vectors[svi]
			synced = [[-1, -1]]
			# Choose
			if synced is not None:
				if sv[0] != -1:
					if len(trans_Choose[sv[0]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][0] = trans_Choose[sv[0]][0]
						for i in range(1, len(trans_Choose[sv[0]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][0] = trans_Choose[sv[0]][i]
			if synced is not None:
				for sync in synced:
					sync[-1] = sv[-1]
					sync.append(svi)
				transitions.extend(filter(lambda s: s[-2] != -1, synced))
		# Convert to Transition instances
		for i in range(len(transitions)):
			transitions[i] = Transition(transitions[i][-1], transitions[i][-2], transitions[i])
			del transitions[i].transitions[-1]
			del transitions[i].transitions[-1]
		# Done
		return transitions
	
	def get_rate(self, state: State, transition: Transition) -> Optional[float]:
		for i in range(len(self.components)):
			if transition.transitions[i] != -1:
				rate = self.components[i].get_rate_value(state, transition.transitions[i])
				if rate is not None:
					for j in range(i + 1, len(self.components)):
						if transition.transitions[j] != -1:
							check_rate = self.components[j].get_rate_value(state, transition)
							if check_rate is not None:
								raise ExplorationError("Invalid MA model: Multiple components specify a rate for the same transition.")
					return rate
		return None
	
	def get_branches(self, state: State, transition: Transition) -> List[Branch]:
		combs = [[-1]]
		probs = [1.0]
		if transition.transitions[0] != -1:
			existing = len(combs)
			branch_count = self._aut_Choose.get_branch_count(state, transition.transitions[0])
			for i in range(1, branch_count):
				probability = self._aut_Choose.get_probability_value(state, transition.transitions[0], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][0] = i
					probs.append(probs[j] * probability)
			probability = self._aut_Choose.get_probability_value(state, transition.transitions[0], 0)
			for i in range(existing):
				combs[i][0] = 0
				probs[i] *= probability
		# Convert to Branch instances
		for i in range(len(combs)):
			combs[i] = Branch(probs[i], combs[i])
		# Done
		return list(filter(lambda b: b.probability > 0.0, combs))
	
	def jump(self, state: State, transition: Transition, branch: Branch, expressions: List[int] = []) -> State:
		transient = self._get_initial_transient()
		for i in range(0, 1):
			target_state = State()
			state.copy_to(target_state)
			target_transient = Transient()
			transient.copy_to(target_transient)
			if transition.transitions[0] != -1:
				self._aut_Choose.jump(state, transient, transition.transitions[0], branch.branches[0], i, target_state, target_transient)
			state = target_state
			transient = target_transient
		for i in range(len(expressions)):
			expressions[i] = self._get_jump_expression_value(state, transient, expressions[i])
		return state
	
	def jump_np(self, state: State, transition: Transition, expressions: List[int] = []) -> State:
		return self.jump(state, transition, self.get_branches(state, transition)[0], expressions)
