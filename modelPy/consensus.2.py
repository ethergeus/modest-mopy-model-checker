# consensus.2

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
	__slots__ = ("counter", "pc1", "coin1", "pc2", "coin2", "process1_location", "process2_location")
	
	def get_variable_value(self, variable: int):
		if variable == 0:
			return self.counter
		elif variable == 1:
			return self.pc1
		elif variable == 2:
			return self.coin1
		elif variable == 3:
			return self.pc2
		elif variable == 4:
			return self.coin2
		elif variable == 5:
			return self.process1_location
		elif variable == 6:
			return self.process2_location
	
	def copy_to(self, other: State):
		other.counter = self.counter
		other.pc1 = self.pc1
		other.coin1 = self.coin1
		other.pc2 = self.pc2
		other.coin2 = self.coin2
		other.process1_location = self.process1_location
		other.process2_location = self.process2_location
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.counter == other.counter and self.pc1 == other.pc1 and self.coin1 == other.coin1 and self.pc2 == other.pc2 and self.coin2 == other.coin2 and self.process1_location == other.process1_location and self.process2_location == other.process2_location
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.counter)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc2)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin2)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.process1_location)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.process2_location)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "counter = " + str(self.counter)
		result += ", pc1 = " + str(self.pc1)
		result += ", coin1 = " + str(self.coin1)
		result += ", pc2 = " + str(self.pc2)
		result += ", coin2 = " + str(self.coin2)
		result += ", process1_location = " + str(self.process1_location)
		result += ", process2_location = " + str(self.process2_location)
		result += ")"
		return result

# Transients
class Transient(object):
	__slots__ = ("finished", "all_coins_equal_0", "all_coins_equal_1", "agree", "steps", "exit_reward_0")
	
	def copy_to(self, other: Transient):
		other.finished = self.finished
		other.all_coins_equal_0 = self.all_coins_equal_0
		other.all_coins_equal_1 = self.all_coins_equal_1
		other.agree = self.agree
		other.steps = self.steps
		other.exit_reward_0 = self.exit_reward_0
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.finished == other.finished and self.all_coins_equal_0 == other.all_coins_equal_0 and self.all_coins_equal_1 == other.all_coins_equal_1 and self.agree == other.agree and self.steps == other.steps and self.exit_reward_0 == other.exit_reward_0
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.finished)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.all_coins_equal_0)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.all_coins_equal_1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.agree)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.steps)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.exit_reward_0)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "finished = " + str(self.finished)
		result += ", all_coins_equal_0 = " + str(self.all_coins_equal_0)
		result += ", all_coins_equal_1 = " + str(self.all_coins_equal_1)
		result += ", agree = " + str(self.agree)
		result += ", steps = " + str(self.steps)
		result += ", exit_reward_0 = " + str(self.exit_reward_0)
		result += ")"
		return result

# Automaton: process1
class process1Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [2, 3, 4, 2]
		self.transition_labels = [[2, 3], [2, 2, 3], [2, 2, 2, 3], [1, 3]]
		self.branch_counts = [[2, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		state.process1_location = 0
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = state.process1_location
		if location == 0:
			if transient_variable == "finished":
				return False
			elif transient_variable == "all_coins_equal_0":
				return ((state.coin1 == 0) and (state.coin2 == 0))
			elif transient_variable == "all_coins_equal_1":
				return ((state.coin1 == 1) and (state.coin2 == 1))
			elif transient_variable == "agree":
				return (state.coin1 == state.coin2)
			elif transient_variable == "steps":
				return 1
		elif location == 1:
			if transient_variable == "finished":
				return False
			elif transient_variable == "all_coins_equal_0":
				return ((state.coin1 == 0) and (state.coin2 == 0))
			elif transient_variable == "all_coins_equal_1":
				return ((state.coin1 == 1) and (state.coin2 == 1))
			elif transient_variable == "agree":
				return (state.coin1 == state.coin2)
			elif transient_variable == "steps":
				return 1
		elif location == 2:
			if transient_variable == "finished":
				return False
			elif transient_variable == "all_coins_equal_0":
				return ((state.coin1 == 0) and (state.coin2 == 0))
			elif transient_variable == "all_coins_equal_1":
				return ((state.coin1 == 1) and (state.coin2 == 1))
			elif transient_variable == "agree":
				return (state.coin1 == state.coin2)
			elif transient_variable == "steps":
				return 1
		elif location == 3:
			if transient_variable == "finished":
				return (state.pc2 == 3)
			elif transient_variable == "all_coins_equal_0":
				return ((state.coin1 == 0) and (state.coin2 == 0))
			elif transient_variable == "all_coins_equal_1":
				return ((state.coin1 == 1) and (state.coin2 == 1))
			elif transient_variable == "agree":
				return (state.coin1 == state.coin2)
			elif transient_variable == "steps":
				return 1
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[state.process1_location]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[state.process1_location][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = state.process1_location
		if location == 0:
			return True
		elif location == 1:
			if transition == 0:
				return ((state.coin1 == 0) and (state.counter > 0))
			elif transition == 1:
				return ((state.coin1 == 1) and (state.counter < 12))
			elif transition == 2:
				return True
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return (state.counter <= 2)
			elif transition == 1:
				return (state.counter >= 10)
			elif transition == 2:
				return ((state.counter > 2) and (state.counter < 10))
			elif transition == 3:
				return True
			else:
				raise IndexError
		elif location == 3:
			return True
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = state.process1_location
		if location == 0:
			return None
		elif location == 1:
			return None
		elif location == 2:
			return None
		elif location == 3:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[state.process1_location][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = state.process1_location
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			else:
				raise IndexError
		elif location == 3:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -3:
			location = state.process1_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 1
					elif branch == 1:
						target_transient.steps = 1
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 1
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 2:
					if branch == 0:
						target_transient.steps = 1
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 2:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 3:
					if branch == 0:
						target_transient.steps = 1
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 1
		elif assignment_index == -2:
			location = state.process1_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == -1:
			location = state.process1_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 0
					elif branch == 1:
						target_transient.steps = 0
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 0
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 2:
					if branch == 0:
						target_transient.steps = 0
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 2:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 3:
					if branch == 0:
						target_transient.steps = 0
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 1:
					if branch == 0:
						target_transient.steps = 0
		elif assignment_index == 0:
			location = state.process1_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc1 = 1
						target_state.coin1 = 0
					elif branch == 1:
						target_state.pc1 = 1
						target_state.coin1 = 1
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc1 = 2
						target_state.coin1 = 0
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 12:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 12 for variable \"counter\".")
						target_state.pc1 = 2
						target_state.coin1 = 0
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_state.pc1 = 3
						target_state.coin1 = 0
				elif transition == 1:
					if branch == 0:
						target_state.pc1 = 3
						target_state.coin1 = 1
				elif transition == 2:
					if branch == 0:
						target_state.pc1 = 0
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_state.pc1 = 3
		elif assignment_index == 1:
			location = state.process1_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 1
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 1
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 0
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 2
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 2
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 1
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 3
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 3
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 0
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 2
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 3
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process1_location = 3

# Automaton: process2
class process2Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [2, 3, 4, 2]
		self.transition_labels = [[2, 3], [2, 2, 3], [2, 2, 2, 3], [1, 3]]
		self.branch_counts = [[2, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		state.process2_location = 0
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = state.process2_location
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[state.process2_location]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[state.process2_location][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = state.process2_location
		if location == 0:
			return True
		elif location == 1:
			if transition == 0:
				return ((state.coin2 == 0) and (state.counter > 0))
			elif transition == 1:
				return ((state.coin2 == 1) and (state.counter < 12))
			elif transition == 2:
				return True
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return (state.counter <= 2)
			elif transition == 1:
				return (state.counter >= 10)
			elif transition == 2:
				return ((state.counter > 2) and (state.counter < 10))
			elif transition == 3:
				return True
			else:
				raise IndexError
		elif location == 3:
			return True
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = state.process2_location
		if location == 0:
			return None
		elif location == 1:
			return None
		elif location == 2:
			return None
		elif location == 3:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[state.process2_location][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = state.process2_location
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			else:
				raise IndexError
		elif location == 3:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = state.process2_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = state.process2_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc2 = 1
						target_state.coin2 = 0
					elif branch == 1:
						target_state.pc2 = 1
						target_state.coin2 = 1
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc2 = 2
						target_state.coin2 = 0
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 12:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 12 for variable \"counter\".")
						target_state.pc2 = 2
						target_state.coin2 = 0
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_state.pc2 = 3
						target_state.coin2 = 0
				elif transition == 1:
					if branch == 0:
						target_state.pc2 = 3
						target_state.coin2 = 1
				elif transition == 2:
					if branch == 0:
						target_state.pc2 = 0
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_state.pc2 = 3
		elif assignment_index == 1:
			location = state.process2_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 1
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 1
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 0
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 2
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 2
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 1
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 3
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 3
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 0
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 2
			elif location == 3:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 3
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
						target_state.process2_location = 3

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
	
	def __init__(self, sync_vector: int, label: int = 0, transitions: List[int] = [-1, -1]):
		self.sync_vector = sync_vector
		self.label = label
		self.transitions = transitions

class Branch(object):
	__slots__ = ("probability", "branches")
	
	def __init__(self, probability = 0.0, branches = [0, 0]):
		self.probability = probability
		self.branches = branches

class Network(object):
	__slots__ = ("network", "model_type", "components", "transition_labels", "sync_vectors", "properties", "variables", "_initial_transient", "_aut_process1", "_aut_process2")
	
	def __init__(self):
		self.network = self
		self.model_type = "mdp"
		self.transition_labels = { 0: "τ", 1: "done", 2: "tau", 3: "set" }
		self.sync_vectors = [[0, -1, 0], [-1, 0, 0], [1, 1, 1], [2, 3, 2], [3, 2, 2]]
		self.properties = [
			Property("c1", PropertyExpression(">=", [PropertyExpression("p_min", [PropertyExpression("eventually", [PropertyExpression("ap", [0])])]), 1.0])),
			Property("c2", PropertyExpression("p_min", [PropertyExpression("eventually", [PropertyExpression("ap", [1])])])),
			Property("disagree", PropertyExpression("p_max", [PropertyExpression("eventually", [PropertyExpression("ap", [2])])])),
			Property("steps_max", PropertyExpression("e_max_s", [3, PropertyExpression("ap", [0])])),
			Property("steps_min", PropertyExpression("e_min_s", [3, PropertyExpression("ap", [0])]))
		]
		self.variables = [
			VariableInfo("counter", None, "int", 0, 12),
			VariableInfo("pc1", None, "int", 0, 3),
			VariableInfo("coin1", None, "int", 0, 1),
			VariableInfo("pc2", None, "int", 0, 3),
			VariableInfo("coin2", None, "int", 0, 1),
			VariableInfo("process1_location", 0, "int", 0, 3),
			VariableInfo("process2_location", 1, "int", 0, 3)
		]
		self._aut_process1 = process1Automaton(self)
		self._aut_process2 = process2Automaton(self)
		self.components = [self._aut_process1, self._aut_process2]
		self._initial_transient = self._get_initial_transient()
	
	def get_initial_state(self) -> State:
		state = State()
		state.counter = 6
		state.pc1 = 0
		state.coin1 = 0
		state.pc2 = 0
		state.coin2 = 0
		self._aut_process1.set_initial_values(state)
		self._aut_process2.set_initial_values(state)
		return state
	
	def _get_initial_transient(self) -> Transient:
		transient = Transient()
		transient.finished = False
		transient.all_coins_equal_0 = False
		transient.all_coins_equal_1 = False
		transient.agree = False
		transient.steps = 0
		transient.exit_reward_0 = 0
		self._aut_process1.set_initial_transient_values(transient)
		self._aut_process2.set_initial_transient_values(transient)
		return transient
	
	def get_expression_value(self, state: State, expression: int):
		if expression == 0:
			return self.network._get_transient_value(state, "finished")
		elif expression == 1:
			return (self.network._get_transient_value(state, "finished") and self.network._get_transient_value(state, "all_coins_equal_1"))
		elif expression == 2:
			return (self.network._get_transient_value(state, "finished") and (not self.network._get_transient_value(state, "agree")))
		elif expression == 3:
			return (self.network._get_transient_value(state, "steps") + self.network._get_transient_value(state, "exit_reward_0"))
		else:
			raise IndexError
	
	def _get_jump_expression_value(self, state: State, transient: Transient, expression: int):
		if expression == 0:
			return transient.finished
		elif expression == 1:
			return (transient.finished and transient.all_coins_equal_1)
		elif expression == 2:
			return (transient.finished and (not transient.agree))
		elif expression == 3:
			return (transient.steps + transient.exit_reward_0)
		else:
			raise IndexError
	
	def _get_transient_value(self, state: State, transient_variable: str):
		# Query the automata for the current value of the transient variable
		result = self._aut_process1.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process2.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		# No automaton has a value: return the transient variable's (cached) initial value
		return getattr(self._initial_transient, transient_variable)
	
	def get_transitions(self, state: State) -> List[Transition]:
		# Collect all automaton transitions, gathered by label
		transitions = []
		trans_process1 = [[], [], [], []]
		transition_count = self._aut_process1.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process1.get_guard_value(state, i):
				trans_process1[self._aut_process1.get_transition_label(state, i)].append(i)
		trans_process2 = [[], [], [], []]
		transition_count = self._aut_process2.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process2.get_guard_value(state, i):
				trans_process2[self._aut_process2.get_transition_label(state, i)].append(i)
		# Match automaton transitions onto synchronisation vectors
		for svi in range(len(self.sync_vectors)):
			sv = self.sync_vectors[svi]
			synced = [[-1, -1, -1]]
			# process1
			if synced is not None:
				if sv[0] != -1:
					if len(trans_process1[sv[0]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][0] = trans_process1[sv[0]][0]
						for i in range(1, len(trans_process1[sv[0]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][0] = trans_process1[sv[0]][i]
			# process2
			if synced is not None:
				if sv[1] != -1:
					if len(trans_process2[sv[1]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][1] = trans_process2[sv[1]][0]
						for i in range(1, len(trans_process2[sv[1]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][1] = trans_process2[sv[1]][i]
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
		combs = [[-1, -1]]
		probs = [1.0]
		if transition.transitions[0] != -1:
			existing = len(combs)
			branch_count = self._aut_process1.get_branch_count(state, transition.transitions[0])
			for i in range(1, branch_count):
				probability = self._aut_process1.get_probability_value(state, transition.transitions[0], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][0] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process1.get_probability_value(state, transition.transitions[0], 0)
			for i in range(existing):
				combs[i][0] = 0
				probs[i] *= probability
		if transition.transitions[1] != -1:
			existing = len(combs)
			branch_count = self._aut_process2.get_branch_count(state, transition.transitions[1])
			for i in range(1, branch_count):
				probability = self._aut_process2.get_probability_value(state, transition.transitions[1], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][1] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process2.get_probability_value(state, transition.transitions[1], 0)
			for i in range(existing):
				combs[i][1] = 0
				probs[i] *= probability
		# Convert to Branch instances
		for i in range(len(combs)):
			combs[i] = Branch(probs[i], combs[i])
		# Done
		result = list(filter(lambda b: b.probability > 0.0, combs))
		if len(result) == 0:
			raise ExplorationError("Invalid model: All branches of a transition have probability zero.")
		return result
	
	def jump(self, state: State, transition: Transition, branch: Branch, expressions: List[int] = []) -> State:
		transient = self._get_initial_transient()
		for i in range(-3, 2):
			target_state = State()
			state.copy_to(target_state)
			target_transient = Transient()
			transient.copy_to(target_transient)
			if transition.transitions[0] != -1:
				self._aut_process1.jump(state, transient, transition.transitions[0], branch.branches[0], i, target_state, target_transient)
			if transition.transitions[1] != -1:
				self._aut_process2.jump(state, transient, transition.transitions[1], branch.branches[1], i, target_state, target_transient)
			state = target_state
			transient = target_transient
		for i in range(len(expressions)):
			expressions[i] = self._get_jump_expression_value(state, transient, expressions[i])
		return state
	
	def jump_np(self, state: State, transition: Transition, expressions: List[int] = []) -> State:
		return self.jump(state, transition, self.get_branches(state, transition)[0], expressions)
