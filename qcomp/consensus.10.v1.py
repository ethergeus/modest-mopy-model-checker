# consensus.10

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
	__slots__ = ("counter", "pc1", "coin1", "pc2", "coin2", "pc3", "coin3", "pc4", "coin4", "pc5", "coin5", "pc6", "coin6", "pc7", "coin7", "pc8", "coin8", "pc9", "coin9", "pc10", "coin10")
	
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
			return self.pc3
		elif variable == 6:
			return self.coin3
		elif variable == 7:
			return self.pc4
		elif variable == 8:
			return self.coin4
		elif variable == 9:
			return self.pc5
		elif variable == 10:
			return self.coin5
		elif variable == 11:
			return self.pc6
		elif variable == 12:
			return self.coin6
		elif variable == 13:
			return self.pc7
		elif variable == 14:
			return self.coin7
		elif variable == 15:
			return self.pc8
		elif variable == 16:
			return self.coin8
		elif variable == 17:
			return self.pc9
		elif variable == 18:
			return self.coin9
		elif variable == 19:
			return self.pc10
		elif variable == 20:
			return self.coin10
	
	def copy_to(self, other: State):
		other.counter = self.counter
		other.pc1 = self.pc1
		other.coin1 = self.coin1
		other.pc2 = self.pc2
		other.coin2 = self.coin2
		other.pc3 = self.pc3
		other.coin3 = self.coin3
		other.pc4 = self.pc4
		other.coin4 = self.coin4
		other.pc5 = self.pc5
		other.coin5 = self.coin5
		other.pc6 = self.pc6
		other.coin6 = self.coin6
		other.pc7 = self.pc7
		other.coin7 = self.coin7
		other.pc8 = self.pc8
		other.coin8 = self.coin8
		other.pc9 = self.pc9
		other.coin9 = self.coin9
		other.pc10 = self.pc10
		other.coin10 = self.coin10
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.counter == other.counter and self.pc1 == other.pc1 and self.coin1 == other.coin1 and self.pc2 == other.pc2 and self.coin2 == other.coin2 and self.pc3 == other.pc3 and self.coin3 == other.coin3 and self.pc4 == other.pc4 and self.coin4 == other.coin4 and self.pc5 == other.pc5 and self.coin5 == other.coin5 and self.pc6 == other.pc6 and self.coin6 == other.coin6 and self.pc7 == other.pc7 and self.coin7 == other.coin7 and self.pc8 == other.pc8 and self.coin8 == other.coin8 and self.pc9 == other.pc9 and self.coin9 == other.coin9 and self.pc10 == other.pc10 and self.coin10 == other.coin10
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.counter)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc2)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin2)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc3)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin3)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc4)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin4)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc5)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin5)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc6)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin6)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc7)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin7)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc8)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin8)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc9)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin9)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.pc10)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.coin10)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "counter = " + str(self.counter)
		result += ", pc1 = " + str(self.pc1)
		result += ", coin1 = " + str(self.coin1)
		result += ", pc2 = " + str(self.pc2)
		result += ", coin2 = " + str(self.coin2)
		result += ", pc3 = " + str(self.pc3)
		result += ", coin3 = " + str(self.coin3)
		result += ", pc4 = " + str(self.pc4)
		result += ", coin4 = " + str(self.coin4)
		result += ", pc5 = " + str(self.pc5)
		result += ", coin5 = " + str(self.coin5)
		result += ", pc6 = " + str(self.pc6)
		result += ", coin6 = " + str(self.coin6)
		result += ", pc7 = " + str(self.pc7)
		result += ", coin7 = " + str(self.coin7)
		result += ", pc8 = " + str(self.pc8)
		result += ", coin8 = " + str(self.coin8)
		result += ", pc9 = " + str(self.pc9)
		result += ", coin9 = " + str(self.coin9)
		result += ", pc10 = " + str(self.pc10)
		result += ", coin10 = " + str(self.coin10)
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
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		if location == 0:
			if transient_variable == "finished":
				return (((((state.pc10 == 3) and (state.pc2 == 3)) and ((state.pc6 == 3) and (state.pc5 == 3))) and ((state.pc8 == 3) and (state.pc3 == 3))) and (((state.pc9 == 3) and (state.pc1 == 3)) and ((state.pc7 == 3) and (state.pc4 == 3))))
			elif transient_variable == "all_coins_equal_0":
				return (((((state.coin10 == 0) and (state.coin2 == 0)) and ((state.coin6 == 0) and (state.coin5 == 0))) and ((state.coin8 == 0) and (state.coin3 == 0))) and (((state.coin9 == 0) and (state.coin1 == 0)) and ((state.coin7 == 0) and (state.coin4 == 0))))
			elif transient_variable == "all_coins_equal_1":
				return (((((state.coin10 == 1) and (state.coin2 == 1)) and ((state.coin6 == 1) and (state.coin5 == 1))) and ((state.coin8 == 1) and (state.coin3 == 1))) and (((state.coin9 == 1) and (state.coin1 == 1)) and ((state.coin7 == 1) and (state.coin4 == 1))))
			elif transient_variable == "agree":
				return (((((state.coin9 == state.coin10) and (state.coin2 == state.coin3)) and (state.coin5 == state.coin6)) and ((state.coin7 == state.coin8) and (state.coin3 == state.coin4))) and (((state.coin8 == state.coin9) and (state.coin1 == state.coin2)) and ((state.coin6 == state.coin7) and (state.coin4 == state.coin5))))
			elif transient_variable == "steps":
				return 1
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc1 == 0)
			elif transition == 1:
				return (((state.pc1 == 1) and (state.coin1 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc1 == 1) and (state.coin1 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc1 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc1 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc1 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc1 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -3:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 1
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 5:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 6:
					if branch == 0:
						target_transient.steps = 1
				elif transition == 7:
					if branch == 0:
						target_transient.steps = 1
		elif assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == -1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.steps = 0
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 5:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 6:
					if branch == 0:
						target_transient.steps = 0
				elif transition == 7:
					if branch == 0:
						target_transient.steps = 0
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc1 = 1
						target_state.coin1 = 0
					elif branch == 1:
						target_state.pc1 = 1
						target_state.coin1 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc1 = 2
						target_state.coin1 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc1 = 2
						target_state.coin1 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc1 = 3
						target_state.coin1 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc1 = 3
						target_state.coin1 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc1 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc1 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process2
class process2Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc2 == 0)
			elif transition == 1:
				return (((state.pc2 == 1) and (state.coin2 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc2 == 1) and (state.coin2 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc2 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc2 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc2 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc2 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc2 = 1
						target_state.coin2 = 0
					elif branch == 1:
						target_state.pc2 = 1
						target_state.coin2 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc2 = 2
						target_state.coin2 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc2 = 2
						target_state.coin2 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc2 = 3
						target_state.coin2 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc2 = 3
						target_state.coin2 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc2 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc2 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process3
class process3Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc3 == 0)
			elif transition == 1:
				return (((state.pc3 == 1) and (state.coin3 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc3 == 1) and (state.coin3 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc3 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc3 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc3 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc3 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc3 = 1
						target_state.coin3 = 0
					elif branch == 1:
						target_state.pc3 = 1
						target_state.coin3 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc3 = 2
						target_state.coin3 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc3 = 2
						target_state.coin3 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc3 = 3
						target_state.coin3 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc3 = 3
						target_state.coin3 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc3 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc3 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process4
class process4Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc4 == 0)
			elif transition == 1:
				return (((state.pc4 == 1) and (state.coin4 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc4 == 1) and (state.coin4 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc4 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc4 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc4 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc4 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc4 = 1
						target_state.coin4 = 0
					elif branch == 1:
						target_state.pc4 = 1
						target_state.coin4 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc4 = 2
						target_state.coin4 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc4 = 2
						target_state.coin4 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc4 = 3
						target_state.coin4 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc4 = 3
						target_state.coin4 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc4 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc4 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process5
class process5Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc5 == 0)
			elif transition == 1:
				return (((state.pc5 == 1) and (state.coin5 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc5 == 1) and (state.coin5 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc5 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc5 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc5 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc5 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc5 = 1
						target_state.coin5 = 0
					elif branch == 1:
						target_state.pc5 = 1
						target_state.coin5 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc5 = 2
						target_state.coin5 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc5 = 2
						target_state.coin5 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc5 = 3
						target_state.coin5 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc5 = 3
						target_state.coin5 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc5 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc5 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process6
class process6Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc6 == 0)
			elif transition == 1:
				return (((state.pc6 == 1) and (state.coin6 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc6 == 1) and (state.coin6 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc6 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc6 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc6 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc6 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc6 = 1
						target_state.coin6 = 0
					elif branch == 1:
						target_state.pc6 = 1
						target_state.coin6 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc6 = 2
						target_state.coin6 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc6 = 2
						target_state.coin6 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc6 = 3
						target_state.coin6 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc6 = 3
						target_state.coin6 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc6 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc6 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process7
class process7Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc7 == 0)
			elif transition == 1:
				return (((state.pc7 == 1) and (state.coin7 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc7 == 1) and (state.coin7 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc7 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc7 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc7 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc7 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc7 = 1
						target_state.coin7 = 0
					elif branch == 1:
						target_state.pc7 = 1
						target_state.coin7 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc7 = 2
						target_state.coin7 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc7 = 2
						target_state.coin7 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc7 = 3
						target_state.coin7 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc7 = 3
						target_state.coin7 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc7 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc7 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process8
class process8Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc8 == 0)
			elif transition == 1:
				return (((state.pc8 == 1) and (state.coin8 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc8 == 1) and (state.coin8 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc8 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc8 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc8 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc8 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc8 = 1
						target_state.coin8 = 0
					elif branch == 1:
						target_state.pc8 = 1
						target_state.coin8 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc8 = 2
						target_state.coin8 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc8 = 2
						target_state.coin8 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc8 = 3
						target_state.coin8 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc8 = 3
						target_state.coin8 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc8 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc8 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process9
class process9Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc9 == 0)
			elif transition == 1:
				return (((state.pc9 == 1) and (state.coin9 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc9 == 1) and (state.coin9 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc9 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc9 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc9 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc9 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc9 = 1
						target_state.coin9 = 0
					elif branch == 1:
						target_state.pc9 = 1
						target_state.coin9 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc9 = 2
						target_state.coin9 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc9 = 2
						target_state.coin9 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc9 = 3
						target_state.coin9 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc9 = 3
						target_state.coin9 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc9 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc9 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

# Automaton: process10
class process10Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [8]
		self.transition_labels = [[2, 2, 2, 2, 2, 2, 1, 3]]
		self.branch_counts = [[2, 1, 1, 1, 1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		pass
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = 0
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[0]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[0][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = 0
		if location == 0:
			if transition == 0:
				return (state.pc10 == 0)
			elif transition == 1:
				return (((state.pc10 == 1) and (state.coin10 == 0)) and (state.counter > 0))
			elif transition == 2:
				return (((state.pc10 == 1) and (state.coin10 == 1)) and (state.counter < 60))
			elif transition == 3:
				return ((state.pc10 == 2) and (state.counter <= 10))
			elif transition == 4:
				return ((state.pc10 == 2) and (state.counter >= 50))
			elif transition == 5:
				return (((state.pc10 == 2) and (state.counter > 10)) and (state.counter < 50))
			elif transition == 6:
				return (state.pc10 == 3)
			elif transition == 7:
				return True
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = 0
		if location == 0:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[0][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = 0
		if location == 0:
			if transition == 0:
				if True:
					return (5 / 10)
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
				return 1
			elif transition == 5:
				return 1
			elif transition == 6:
				return 1
			elif transition == 7:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == -2:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
					elif branch == 1:
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
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = transient.steps
		elif assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.pc10 = 1
						target_state.coin10 = 0
					elif branch == 1:
						target_state.pc10 = 1
						target_state.coin10 = 1
				elif transition == 1:
					if branch == 0:
						target_state.counter = (state.counter - 1)
						if target_state.counter < 0:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is less than the lower bound of 0 for variable \"counter\".")
						target_state.pc10 = 2
						target_state.coin10 = 0
				elif transition == 2:
					if branch == 0:
						target_state.counter = (state.counter + 1)
						if target_state.counter > 60:
							raise OverflowError("Assigned value of " + str(target_state.counter) + " is greater than the upper bound of 60 for variable \"counter\".")
						target_state.pc10 = 2
						target_state.coin10 = 0
				elif transition == 3:
					if branch == 0:
						target_state.pc10 = 3
						target_state.coin10 = 0
				elif transition == 4:
					if branch == 0:
						target_state.pc10 = 3
						target_state.coin10 = 1
				elif transition == 5:
					if branch == 0:
						target_state.pc10 = 0
				elif transition == 6:
					if branch == 0:
						target_state.pc10 = 3
		elif assignment_index == 1:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
					elif branch == 1:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 1:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 2:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 3:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 4:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 5:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 6:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)
				elif transition == 7:
					if branch == 0:
						target_transient.exit_reward_0 = (transient.exit_reward_0 - transient.steps)

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
	
	def __init__(self, sync_vector: int, label: int = 0, transitions: List[int] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]):
		self.sync_vector = sync_vector
		self.label = label
		self.transitions = transitions

class Branch(object):
	__slots__ = ("probability", "branches")
	
	def __init__(self, probability = 0.0, branches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
		self.probability = probability
		self.branches = branches

class Network(object):
	__slots__ = ("network", "model_type", "components", "transition_labels", "sync_vectors", "properties", "variables", "_initial_transient", "_aut_process1", "_aut_process2", "_aut_process3", "_aut_process4", "_aut_process5", "_aut_process6", "_aut_process7", "_aut_process8", "_aut_process9", "_aut_process10")
	
	def __init__(self):
		self.network = self
		self.model_type = "mdp"
		self.transition_labels = { 0: "τ", 1: "done", 2: "tau", 3: "set" }
		self.sync_vectors = [[0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0], [-1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0], [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0], [-1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0], [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, 0], [-1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0], [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2], [3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2], [3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2], [3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2], [3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2], [3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2], [3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2], [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2], [3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2], [3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2]]
		self.properties = [
			Property("c1", PropertyExpression(">=", [PropertyExpression("p_min", [PropertyExpression("eventually", [PropertyExpression("ap", [0])])]), 1.0])),
			Property("c2", PropertyExpression("p_min", [PropertyExpression("eventually", [PropertyExpression("ap", [1])])])),
			Property("disagree", PropertyExpression("p_max", [PropertyExpression("eventually", [PropertyExpression("ap", [2])])])),
			Property("steps_max", PropertyExpression("e_max_s", [3, PropertyExpression("ap", [0])])),
			Property("steps_min", PropertyExpression("e_min_s", [3, PropertyExpression("ap", [0])]))
		]
		self.variables = [
			VariableInfo("counter", None, "int", 0, 60),
			VariableInfo("pc1", None, "int", 0, 3),
			VariableInfo("coin1", None, "int", 0, 1),
			VariableInfo("pc2", None, "int", 0, 3),
			VariableInfo("coin2", None, "int", 0, 1),
			VariableInfo("pc3", None, "int", 0, 3),
			VariableInfo("coin3", None, "int", 0, 1),
			VariableInfo("pc4", None, "int", 0, 3),
			VariableInfo("coin4", None, "int", 0, 1),
			VariableInfo("pc5", None, "int", 0, 3),
			VariableInfo("coin5", None, "int", 0, 1),
			VariableInfo("pc6", None, "int", 0, 3),
			VariableInfo("coin6", None, "int", 0, 1),
			VariableInfo("pc7", None, "int", 0, 3),
			VariableInfo("coin7", None, "int", 0, 1),
			VariableInfo("pc8", None, "int", 0, 3),
			VariableInfo("coin8", None, "int", 0, 1),
			VariableInfo("pc9", None, "int", 0, 3),
			VariableInfo("coin9", None, "int", 0, 1),
			VariableInfo("pc10", None, "int", 0, 3),
			VariableInfo("coin10", None, "int", 0, 1)
		]
		self._aut_process1 = process1Automaton(self)
		self._aut_process2 = process2Automaton(self)
		self._aut_process3 = process3Automaton(self)
		self._aut_process4 = process4Automaton(self)
		self._aut_process5 = process5Automaton(self)
		self._aut_process6 = process6Automaton(self)
		self._aut_process7 = process7Automaton(self)
		self._aut_process8 = process8Automaton(self)
		self._aut_process9 = process9Automaton(self)
		self._aut_process10 = process10Automaton(self)
		self.components = [self._aut_process1, self._aut_process2, self._aut_process3, self._aut_process4, self._aut_process5, self._aut_process6, self._aut_process7, self._aut_process8, self._aut_process9, self._aut_process10]
		self._initial_transient = self._get_initial_transient()
	
	def get_initial_state(self) -> State:
		state = State()
		state.counter = 30
		state.pc1 = 0
		state.coin1 = 0
		state.pc2 = 0
		state.coin2 = 0
		state.pc3 = 0
		state.coin3 = 0
		state.pc4 = 0
		state.coin4 = 0
		state.pc5 = 0
		state.coin5 = 0
		state.pc6 = 0
		state.coin6 = 0
		state.pc7 = 0
		state.coin7 = 0
		state.pc8 = 0
		state.coin8 = 0
		state.pc9 = 0
		state.coin9 = 0
		state.pc10 = 0
		state.coin10 = 0
		self._aut_process1.set_initial_values(state)
		self._aut_process2.set_initial_values(state)
		self._aut_process3.set_initial_values(state)
		self._aut_process4.set_initial_values(state)
		self._aut_process5.set_initial_values(state)
		self._aut_process6.set_initial_values(state)
		self._aut_process7.set_initial_values(state)
		self._aut_process8.set_initial_values(state)
		self._aut_process9.set_initial_values(state)
		self._aut_process10.set_initial_values(state)
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
		self._aut_process3.set_initial_transient_values(transient)
		self._aut_process4.set_initial_transient_values(transient)
		self._aut_process5.set_initial_transient_values(transient)
		self._aut_process6.set_initial_transient_values(transient)
		self._aut_process7.set_initial_transient_values(transient)
		self._aut_process8.set_initial_transient_values(transient)
		self._aut_process9.set_initial_transient_values(transient)
		self._aut_process10.set_initial_transient_values(transient)
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
		result = self._aut_process3.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process4.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process5.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process6.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process7.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process8.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process9.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_process10.get_transient_value(state, transient_variable)
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
		trans_process3 = [[], [], [], []]
		transition_count = self._aut_process3.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process3.get_guard_value(state, i):
				trans_process3[self._aut_process3.get_transition_label(state, i)].append(i)
		trans_process4 = [[], [], [], []]
		transition_count = self._aut_process4.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process4.get_guard_value(state, i):
				trans_process4[self._aut_process4.get_transition_label(state, i)].append(i)
		trans_process5 = [[], [], [], []]
		transition_count = self._aut_process5.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process5.get_guard_value(state, i):
				trans_process5[self._aut_process5.get_transition_label(state, i)].append(i)
		trans_process6 = [[], [], [], []]
		transition_count = self._aut_process6.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process6.get_guard_value(state, i):
				trans_process6[self._aut_process6.get_transition_label(state, i)].append(i)
		trans_process7 = [[], [], [], []]
		transition_count = self._aut_process7.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process7.get_guard_value(state, i):
				trans_process7[self._aut_process7.get_transition_label(state, i)].append(i)
		trans_process8 = [[], [], [], []]
		transition_count = self._aut_process8.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process8.get_guard_value(state, i):
				trans_process8[self._aut_process8.get_transition_label(state, i)].append(i)
		trans_process9 = [[], [], [], []]
		transition_count = self._aut_process9.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process9.get_guard_value(state, i):
				trans_process9[self._aut_process9.get_transition_label(state, i)].append(i)
		trans_process10 = [[], [], [], []]
		transition_count = self._aut_process10.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_process10.get_guard_value(state, i):
				trans_process10[self._aut_process10.get_transition_label(state, i)].append(i)
		# Match automaton transitions onto synchronisation vectors
		for svi in range(len(self.sync_vectors)):
			sv = self.sync_vectors[svi]
			synced = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
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
			# process3
			if synced is not None:
				if sv[2] != -1:
					if len(trans_process3[sv[2]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][2] = trans_process3[sv[2]][0]
						for i in range(1, len(trans_process3[sv[2]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][2] = trans_process3[sv[2]][i]
			# process4
			if synced is not None:
				if sv[3] != -1:
					if len(trans_process4[sv[3]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][3] = trans_process4[sv[3]][0]
						for i in range(1, len(trans_process4[sv[3]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][3] = trans_process4[sv[3]][i]
			# process5
			if synced is not None:
				if sv[4] != -1:
					if len(trans_process5[sv[4]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][4] = trans_process5[sv[4]][0]
						for i in range(1, len(trans_process5[sv[4]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][4] = trans_process5[sv[4]][i]
			# process6
			if synced is not None:
				if sv[5] != -1:
					if len(trans_process6[sv[5]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][5] = trans_process6[sv[5]][0]
						for i in range(1, len(trans_process6[sv[5]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][5] = trans_process6[sv[5]][i]
			# process7
			if synced is not None:
				if sv[6] != -1:
					if len(trans_process7[sv[6]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][6] = trans_process7[sv[6]][0]
						for i in range(1, len(trans_process7[sv[6]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][6] = trans_process7[sv[6]][i]
			# process8
			if synced is not None:
				if sv[7] != -1:
					if len(trans_process8[sv[7]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][7] = trans_process8[sv[7]][0]
						for i in range(1, len(trans_process8[sv[7]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][7] = trans_process8[sv[7]][i]
			# process9
			if synced is not None:
				if sv[8] != -1:
					if len(trans_process9[sv[8]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][8] = trans_process9[sv[8]][0]
						for i in range(1, len(trans_process9[sv[8]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][8] = trans_process9[sv[8]][i]
			# process10
			if synced is not None:
				if sv[9] != -1:
					if len(trans_process10[sv[9]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][9] = trans_process10[sv[9]][0]
						for i in range(1, len(trans_process10[sv[9]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][9] = trans_process10[sv[9]][i]
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
		combs = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
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
		if transition.transitions[2] != -1:
			existing = len(combs)
			branch_count = self._aut_process3.get_branch_count(state, transition.transitions[2])
			for i in range(1, branch_count):
				probability = self._aut_process3.get_probability_value(state, transition.transitions[2], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][2] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process3.get_probability_value(state, transition.transitions[2], 0)
			for i in range(existing):
				combs[i][2] = 0
				probs[i] *= probability
		if transition.transitions[3] != -1:
			existing = len(combs)
			branch_count = self._aut_process4.get_branch_count(state, transition.transitions[3])
			for i in range(1, branch_count):
				probability = self._aut_process4.get_probability_value(state, transition.transitions[3], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][3] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process4.get_probability_value(state, transition.transitions[3], 0)
			for i in range(existing):
				combs[i][3] = 0
				probs[i] *= probability
		if transition.transitions[4] != -1:
			existing = len(combs)
			branch_count = self._aut_process5.get_branch_count(state, transition.transitions[4])
			for i in range(1, branch_count):
				probability = self._aut_process5.get_probability_value(state, transition.transitions[4], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][4] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process5.get_probability_value(state, transition.transitions[4], 0)
			for i in range(existing):
				combs[i][4] = 0
				probs[i] *= probability
		if transition.transitions[5] != -1:
			existing = len(combs)
			branch_count = self._aut_process6.get_branch_count(state, transition.transitions[5])
			for i in range(1, branch_count):
				probability = self._aut_process6.get_probability_value(state, transition.transitions[5], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][5] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process6.get_probability_value(state, transition.transitions[5], 0)
			for i in range(existing):
				combs[i][5] = 0
				probs[i] *= probability
		if transition.transitions[6] != -1:
			existing = len(combs)
			branch_count = self._aut_process7.get_branch_count(state, transition.transitions[6])
			for i in range(1, branch_count):
				probability = self._aut_process7.get_probability_value(state, transition.transitions[6], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][6] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process7.get_probability_value(state, transition.transitions[6], 0)
			for i in range(existing):
				combs[i][6] = 0
				probs[i] *= probability
		if transition.transitions[7] != -1:
			existing = len(combs)
			branch_count = self._aut_process8.get_branch_count(state, transition.transitions[7])
			for i in range(1, branch_count):
				probability = self._aut_process8.get_probability_value(state, transition.transitions[7], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][7] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process8.get_probability_value(state, transition.transitions[7], 0)
			for i in range(existing):
				combs[i][7] = 0
				probs[i] *= probability
		if transition.transitions[8] != -1:
			existing = len(combs)
			branch_count = self._aut_process9.get_branch_count(state, transition.transitions[8])
			for i in range(1, branch_count):
				probability = self._aut_process9.get_probability_value(state, transition.transitions[8], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][8] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process9.get_probability_value(state, transition.transitions[8], 0)
			for i in range(existing):
				combs[i][8] = 0
				probs[i] *= probability
		if transition.transitions[9] != -1:
			existing = len(combs)
			branch_count = self._aut_process10.get_branch_count(state, transition.transitions[9])
			for i in range(1, branch_count):
				probability = self._aut_process10.get_probability_value(state, transition.transitions[9], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][9] = i
					probs.append(probs[j] * probability)
			probability = self._aut_process10.get_probability_value(state, transition.transitions[9], 0)
			for i in range(existing):
				combs[i][9] = 0
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
			if transition.transitions[2] != -1:
				self._aut_process3.jump(state, transient, transition.transitions[2], branch.branches[2], i, target_state, target_transient)
			if transition.transitions[3] != -1:
				self._aut_process4.jump(state, transient, transition.transitions[3], branch.branches[3], i, target_state, target_transient)
			if transition.transitions[4] != -1:
				self._aut_process5.jump(state, transient, transition.transitions[4], branch.branches[4], i, target_state, target_transient)
			if transition.transitions[5] != -1:
				self._aut_process6.jump(state, transient, transition.transitions[5], branch.branches[5], i, target_state, target_transient)
			if transition.transitions[6] != -1:
				self._aut_process7.jump(state, transient, transition.transitions[6], branch.branches[6], i, target_state, target_transient)
			if transition.transitions[7] != -1:
				self._aut_process8.jump(state, transient, transition.transitions[7], branch.branches[7], i, target_state, target_transient)
			if transition.transitions[8] != -1:
				self._aut_process9.jump(state, transient, transition.transitions[8], branch.branches[8], i, target_state, target_transient)
			if transition.transitions[9] != -1:
				self._aut_process10.jump(state, transient, transition.transitions[9], branch.branches[9], i, target_state, target_transient)
			state = target_state
			transient = target_transient
		for i in range(len(expressions)):
			expressions[i] = self._get_jump_expression_value(state, transient, expressions[i])
		return state
	
	def jump_np(self, state: State, transition: Transition, expressions: List[int] = []) -> State:
		return self.jump(state, transition, self.get_branches(state, transition)[0], expressions)
