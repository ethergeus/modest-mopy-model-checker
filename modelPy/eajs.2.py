# eajs.2

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
	__slots__ = ("loc_2", "t_2", "boost_1", "user_1", "t_1", "loc_1", "failure_1", "battery_load", "Process_2_location", "Resources_location", "Process_1_location")
	
	def udfs_global_noLocalFailure_state(self):
		return (not (((self.loc_1 == 1) and (self.t_1 == 0)) and self.failure_1))
	
	def udfs_global_noLocalFailure_jump(self, transient):
		return (not (((self.loc_1 == 1) and (self.t_1 == 0)) and self.failure_1))
	
	def udfs_global_emptyBattery_state(self):
		return (self.battery_load == 0)
	
	def udfs_global_emptyBattery_jump(self, transient):
		return (self.battery_load == 0)
	
	def udfs_global_process_2_finishes_state(self):
		return ((self.loc_2 == 1) and (self.t_2 == 0))
	
	def udfs_global_process_2_finishes_jump(self, transient):
		return ((self.loc_2 == 1) and (self.t_2 == 0))
	
	def udfs_global_process_1_finishes_state(self):
		return ((self.loc_1 == 1) and (self.t_1 == 0))
	
	def udfs_global_process_1_finishes_jump(self, transient):
		return ((self.loc_1 == 1) and (self.t_1 == 0))
	
	def udfs_global_localFailure_state(self):
		return (((self.loc_1 == 1) and (self.t_1 == 0)) and self.failure_1)
	
	def udfs_global_localFailure_jump(self, transient):
		return (((self.loc_1 == 1) and (self.t_1 == 0)) and self.failure_1)
	
	def get_variable_value(self, variable: int):
		if variable == 0:
			return self.loc_2
		elif variable == 1:
			return self.t_2
		elif variable == 2:
			return self.boost_1
		elif variable == 3:
			return self.user_1
		elif variable == 4:
			return self.t_1
		elif variable == 5:
			return self.loc_1
		elif variable == 6:
			return self.failure_1
		elif variable == 7:
			return self.battery_load
		elif variable == 8:
			return self.Process_2_location
		elif variable == 9:
			return self.Resources_location
		elif variable == 10:
			return self.Process_1_location
	
	def copy_to(self, other: State):
		other.loc_2 = self.loc_2
		other.t_2 = self.t_2
		other.boost_1 = self.boost_1
		other.user_1 = self.user_1
		other.t_1 = self.t_1
		other.loc_1 = self.loc_1
		other.failure_1 = self.failure_1
		other.battery_load = self.battery_load
		other.Process_2_location = self.Process_2_location
		other.Resources_location = self.Resources_location
		other.Process_1_location = self.Process_1_location
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.loc_2 == other.loc_2 and self.t_2 == other.t_2 and self.boost_1 == other.boost_1 and self.user_1 == other.user_1 and self.t_1 == other.t_1 and self.loc_1 == other.loc_1 and self.failure_1 == other.failure_1 and self.battery_load == other.battery_load and self.Process_2_location == other.Process_2_location and self.Resources_location == other.Resources_location and self.Process_1_location == other.Process_1_location
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.loc_2)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.t_2)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.boost_1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.user_1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.t_1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.loc_1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.failure_1)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.battery_load)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.Process_2_location)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.Resources_location)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.Process_1_location)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "loc_2 = " + str(self.loc_2)
		result += ", t_2 = " + str(self.t_2)
		result += ", boost_1 = " + str(self.boost_1)
		result += ", user_1 = " + str(self.user_1)
		result += ", t_1 = " + str(self.t_1)
		result += ", loc_1 = " + str(self.loc_1)
		result += ", failure_1 = " + str(self.failure_1)
		result += ", battery_load = " + str(self.battery_load)
		result += ", Process_2_location = " + str(self.Process_2_location)
		result += ", Resources_location = " + str(self.Resources_location)
		result += ", Process_1_location = " + str(self.Process_1_location)
		result += ")"
		return result

# Transients
class Transient(object):
	__slots__ = ("label_localFailure", "label_noLocalFailure", "label_emptyBattery", "label_process_1_finishes", "label_process_2_finishes", "energyLocal", "utilityLocal")
	
	def copy_to(self, other: Transient):
		other.label_localFailure = self.label_localFailure
		other.label_noLocalFailure = self.label_noLocalFailure
		other.label_emptyBattery = self.label_emptyBattery
		other.label_process_1_finishes = self.label_process_1_finishes
		other.label_process_2_finishes = self.label_process_2_finishes
		other.energyLocal = self.energyLocal
		other.utilityLocal = self.utilityLocal
	
	def __eq__(self, other):
		return isinstance(other, self.__class__) and self.label_localFailure == other.label_localFailure and self.label_noLocalFailure == other.label_noLocalFailure and self.label_emptyBattery == other.label_emptyBattery and self.label_process_1_finishes == other.label_process_1_finishes and self.label_process_2_finishes == other.label_process_2_finishes and self.energyLocal == other.energyLocal and self.utilityLocal == other.utilityLocal
	
	def __ne__(self, other):
		return not self.__eq__(other)
	
	def __hash__(self):
		result = 75619
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.label_localFailure)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.label_noLocalFailure)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.label_emptyBattery)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.label_process_1_finishes)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.label_process_2_finishes)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.energyLocal)) & 0xFFFFFFFF
		result = (((101 * result) & 0xFFFFFFFF) + hash(self.utilityLocal)) & 0xFFFFFFFF
		return result
	
	def __str__(self):
		result = "("
		result += "label_localFailure = " + str(self.label_localFailure)
		result += ", label_noLocalFailure = " + str(self.label_noLocalFailure)
		result += ", label_emptyBattery = " + str(self.label_emptyBattery)
		result += ", label_process_1_finishes = " + str(self.label_process_1_finishes)
		result += ", label_process_2_finishes = " + str(self.label_process_2_finishes)
		result += ", energyLocal = " + str(self.energyLocal)
		result += ", utilityLocal = " + str(self.utilityLocal)
		result += ")"
		return result

# Automaton: Process_2
class Process_2Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [2, 2, 2]
		self.transition_labels = [[1, 1], [1, 1], [1, 1]]
		self.branch_counts = [[1, 2], [1, 2], [1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		state.Process_2_location = 2
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = state.Process_2_location
		if location == 0:
			if transient_variable == "label_localFailure":
				return (((state.loc_1 == 1) and (state.t_1 == 0)) and state.failure_1)
			elif transient_variable == "label_noLocalFailure":
				return (not (((state.loc_1 == 1) and (state.t_1 == 0)) and state.failure_1))
			elif transient_variable == "label_emptyBattery":
				return (state.battery_load == 0)
			elif transient_variable == "label_process_1_finishes":
				return ((state.loc_1 == 1) and (state.t_1 == 0))
			elif transient_variable == "label_process_2_finishes":
				return False
		elif location == 1:
			if transient_variable == "label_localFailure":
				return (((state.loc_1 == 1) and (state.t_1 == 0)) and state.failure_1)
			elif transient_variable == "label_noLocalFailure":
				return (not (((state.loc_1 == 1) and (state.t_1 == 0)) and state.failure_1))
			elif transient_variable == "label_emptyBattery":
				return (state.battery_load == 0)
			elif transient_variable == "label_process_1_finishes":
				return ((state.loc_1 == 1) and (state.t_1 == 0))
			elif transient_variable == "label_process_2_finishes":
				return (state.t_2 == 0)
		elif location == 2:
			if transient_variable == "label_localFailure":
				return (((state.loc_1 == 1) and (state.t_1 == 0)) and state.failure_1)
			elif transient_variable == "label_noLocalFailure":
				return (not (((state.loc_1 == 1) and (state.t_1 == 0)) and state.failure_1))
			elif transient_variable == "label_emptyBattery":
				return (state.battery_load == 0)
			elif transient_variable == "label_process_1_finishes":
				return ((state.loc_1 == 1) and (state.t_1 == 0))
			elif transient_variable == "label_process_2_finishes":
				return False
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[state.Process_2_location]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[state.Process_2_location][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = state.Process_2_location
		if location == 0:
			if transition == 0:
				return (state.user_1 != 2)
			elif transition == 1:
				return (state.user_1 == 2)
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return (state.t_2 > 0)
			elif transition == 1:
				return (state.t_2 == 0)
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return (state.t_2 != 0)
			elif transition == 1:
				return (state.t_2 == 0)
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = state.Process_2_location
		if location == 0:
			return None
		elif location == 1:
			return None
		elif location == 2:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[state.Process_2_location][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = state.Process_2_location
		if location == 0:
			if transition == 0:
				return 1
			elif transition == 1:
				if branch == 0:
					return (1 / 3)
				elif branch == 1:
					return (2 / 3)
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return 1
			elif transition == 1:
				if branch == 0:
					return (2 / 3)
				elif branch == 1:
					return (1 / 3)
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == 0:
			location = state.Process_2_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.t_2 = max(0, (state.t_2 - 1))
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 0
				elif transition == 1:
					if branch == 0:
						target_state.loc_2 = 1
						target_state.t_2 = 2
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 1
					elif branch == 1:
						target_state.loc_2 = 1
						target_state.t_2 = 3
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 1
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_state.t_2 = max((state.t_2 - (state.boost_1 + 1)), 0)
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 1
				elif transition == 1:
					if branch == 0:
						target_state.loc_2 = 2
						target_state.t_2 = 4
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 2
					elif branch == 1:
						target_state.loc_2 = 2
						target_state.t_2 = 5
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 2
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_state.t_2 = (state.t_2 - 1)
						if target_state.t_2 < 0:
							raise OverflowError("Assigned value of " + str(target_state.t_2) + " is less than the lower bound of 0 for variable \"t_2\".")
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 2
				elif transition == 1:
					if branch == 0:
						target_state.loc_2 = 0
						target_transient.energyLocal = (((9 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1)) else 0) + (1 if (state.loc_1 == 0) else 0)) + ((3 if (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0)) else 0) + (2 if (state.loc_1 == 2) else 0)))
						target_transient.utilityLocal = (1 if (((state.battery_load != 0) and (state.t_1 == 0)) and ((not state.failure_1) and (state.loc_1 == 1))) else 0)
						target_state.Process_2_location = 0

# Automaton: Resources
class ResourcesAutomaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [5, 4, 4]
		self.transition_labels = [[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
		self.branch_counts = [[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
	
	def set_initial_values(self, state: State) -> None:
		state.Resources_location = 0
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = state.Resources_location
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[state.Resources_location]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[state.Resources_location][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = state.Resources_location
		if location == 0:
			if transition >= 0 and transition < 2:
				return (state.loc_1 == 0)
			elif transition >= 2 and transition < 4:
				return (state.loc_2 == 0)
			elif transition == 4:
				return ((state.loc_1 != 0) and (state.loc_2 != 0))
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return (not ((state.loc_1 == 1) and (state.t_1 == 0)))
			elif transition == 1:
				return (((state.loc_2 != 0) and (state.t_1 == 0)) and ((state.loc_1 != 0) and (state.loc_1 == 1)))
			elif transition >= 2:
				return (((state.loc_2 == 0) and (state.t_1 == 0)) and (state.loc_1 == 1))
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return (not ((state.loc_2 == 1) and (state.t_2 == 0)))
			elif transition == 1:
				return (((state.loc_2 != 0) and (state.t_2 == 0)) and ((state.loc_1 != 0) and (state.loc_2 == 1)))
			elif transition >= 2:
				return (((state.loc_1 == 0) and (state.t_2 == 0)) and (state.loc_2 == 1))
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = state.Resources_location
		if location == 0:
			return None
		elif location == 1:
			return None
		elif location == 2:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[state.Resources_location][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = state.Resources_location
		if location == 0:
			if transition == 0:
				return 1
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			elif transition == 4:
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
			elif transition == 3:
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
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == 0:
			location = state.Resources_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.user_1 = 1
						target_state.Resources_location = 1
				elif transition == 1:
					if branch == 0:
						target_state.boost_1 = 1
						target_state.user_1 = 1
						target_state.Resources_location = 1
				elif transition == 2:
					if branch == 0:
						target_state.user_1 = 2
						target_state.Resources_location = 2
				elif transition == 3:
					if branch == 0:
						target_state.boost_1 = 1
						target_state.user_1 = 2
						target_state.Resources_location = 2
				elif transition == 4:
					if branch == 0:
						target_state.Resources_location = 0
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_state.Resources_location = 1
				elif transition == 1:
					if branch == 0:
						target_state.boost_1 = 0
						target_state.user_1 = 0
						target_state.Resources_location = 0
				elif transition == 2:
					if branch == 0:
						target_state.boost_1 = 1
						target_state.user_1 = 2
						target_state.Resources_location = 2
				elif transition == 3:
					if branch == 0:
						target_state.boost_1 = 0
						target_state.user_1 = 2
						target_state.Resources_location = 2
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_state.Resources_location = 2
				elif transition == 1:
					if branch == 0:
						target_state.boost_1 = 0
						target_state.user_1 = 0
						target_state.Resources_location = 0
				elif transition == 2:
					if branch == 0:
						target_state.boost_1 = 1
						target_state.user_1 = 1
						target_state.Resources_location = 1
				elif transition == 3:
					if branch == 0:
						target_state.boost_1 = 0
						target_state.user_1 = 1
						target_state.Resources_location = 1

# Automaton: Process_1
class Process_1Automaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [2, 2, 2]
		self.transition_labels = [[1, 1], [1, 1], [1, 1]]
		self.branch_counts = [[1, 2], [1, 2], [1, 2]]
	
	def set_initial_values(self, state: State) -> None:
		state.Process_1_location = 2
	
	def set_initial_transient_values(self, transient: Transient) -> None:
		pass
	
	def get_transient_value(self, state: State, transient_variable: str):
		location = state.Process_1_location
		return None
	
	def get_transition_count(self, state: State) -> int:
		return self.transition_counts[state.Process_1_location]
	
	def get_transition_label(self, state: State, transition: int) -> int:
		return self.transition_labels[state.Process_1_location][transition]
	
	def get_guard_value(self, state: State, transition: int) -> bool:
		location = state.Process_1_location
		if location == 0:
			if transition == 0:
				return (state.user_1 != 1)
			elif transition == 1:
				return (state.user_1 == 1)
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return (state.t_1 > 0)
			elif transition == 1:
				return (state.t_1 == 0)
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return (state.t_1 != 0)
			elif transition == 1:
				return (state.t_1 == 0)
			else:
				raise IndexError
		else:
			raise IndexError
	
	def get_rate_value(self, state: State, transition: int) -> Optional[float]:
		location = state.Process_1_location
		if location == 0:
			return None
		elif location == 1:
			return None
		elif location == 2:
			return None
		else:
			raise IndexError
	
	def get_branch_count(self, state: State, transition: int) -> int:
		return self.branch_counts[state.Process_1_location][transition]
	
	def get_probability_value(self, state: State, transition: int, branch: int) -> float:
		location = state.Process_1_location
		if location == 0:
			if transition == 0:
				return 1
			elif transition == 1:
				if branch == 0:
					return (1 / 3)
				elif branch == 1:
					return (2 / 3)
			else:
				raise IndexError
		elif location == 1:
			if transition == 0:
				return 1
			elif transition == 1:
				if branch == 0:
					return (2 / 3)
				elif branch == 1:
					return (1 / 3)
			else:
				raise IndexError
		elif location == 2:
			if transition == 0:
				return 1
			elif transition == 1:
				if branch == 0:
					return (2 / 3)
				elif branch == 1:
					return (1 / 3)
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == 0:
			location = state.Process_1_location
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.t_1 = max(0, (state.t_1 - 1))
						target_state.Process_1_location = 0
				elif transition == 1:
					if branch == 0:
						target_state.failure_1 = (2 >= ((state.boost_1 + 1) * state.t_1))
						target_state.t_1 = 2
						target_state.loc_1 = 1
						target_state.Process_1_location = 1
					elif branch == 1:
						target_state.failure_1 = (3 >= ((state.boost_1 + 1) * state.t_1))
						target_state.t_1 = 3
						target_state.loc_1 = 1
						target_state.Process_1_location = 1
			elif location == 1:
				if transition == 0:
					if branch == 0:
						target_state.t_1 = max((state.t_1 - (state.boost_1 + 1)), 0)
						target_state.Process_1_location = 1
				elif transition == 1:
					if branch == 0:
						target_state.failure_1 = False
						target_state.t_1 = 4
						target_state.loc_1 = 2
						target_state.Process_1_location = 2
					elif branch == 1:
						target_state.failure_1 = False
						target_state.t_1 = 5
						target_state.loc_1 = 2
						target_state.Process_1_location = 2
			elif location == 2:
				if transition == 0:
					if branch == 0:
						target_state.t_1 = (state.t_1 - 1)
						if target_state.t_1 < 0:
							raise OverflowError("Assigned value of " + str(target_state.t_1) + " is less than the lower bound of 0 for variable \"t_1\".")
						target_state.Process_1_location = 2
				elif transition == 1:
					if branch == 0:
						target_state.t_1 = 9
						target_state.loc_1 = 0
						target_state.Process_1_location = 0
					elif branch == 1:
						target_state.t_1 = 7
						target_state.loc_1 = 0
						target_state.Process_1_location = 0

# Automaton: Battery
class BatteryAutomaton(object):
	__slots__ = ("network", "transition_counts", "transition_labels", "branch_counts")
	
	def __init__(self, network: Network):
		self.network = network
		self.transition_counts = [4]
		self.transition_labels = [[1, 1, 1, 1]]
		self.branch_counts = [[1, 1, 1, 1]]
	
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
				return (state.loc_1 == 2)
			elif transition == 1:
				return (state.loc_1 == 0)
			elif transition == 2:
				return (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 0))
			elif transition == 3:
				return (((state.loc_1 == 1) and (state.user_1 == 1)) and (state.boost_1 == 1))
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
				return 1
			elif transition == 1:
				return 1
			elif transition == 2:
				return 1
			elif transition == 3:
				return 1
			else:
				raise IndexError
		else:
			raise IndexError
	
	def jump(self, state: State, transient: Transient, transition: int, branch: int, assignment_index: int, target_state: State, target_transient: Transient) -> None:
		if assignment_index == 0:
			location = 0
			if location == 0:
				if transition == 0:
					if branch == 0:
						target_state.battery_load = max(0, (state.battery_load - 2))
				elif transition == 1:
					if branch == 0:
						target_state.battery_load = max(0, (state.battery_load - 1))
				elif transition == 2:
					if branch == 0:
						target_state.battery_load = max(0, (state.battery_load - 3))
				elif transition == 3:
					if branch == 0:
						target_state.battery_load = max(0, (state.battery_load - 9))

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
	
	def __init__(self, sync_vector: int, label: int = 0, transitions: List[int] = [-1, -1, -1, -1]):
		self.sync_vector = sync_vector
		self.label = label
		self.transitions = transitions

class Branch(object):
	__slots__ = ("probability", "branches")
	
	def __init__(self, probability = 0.0, branches = [0, 0, 0, 0]):
		self.probability = probability
		self.branches = branches

class Network(object):
	__slots__ = ("network", "model_type", "components", "transition_labels", "sync_vectors", "properties", "variables", "_initial_transient", "_aut_Process_2", "_aut_Resources", "_aut_Process_1", "_aut_Battery")
	
	def __init__(self):
		self.network = self
		self.model_type = "mdp"
		self.transition_labels = { 0: "τ", 1: "tick" }
		self.sync_vectors = [[0, -1, -1, -1, 0], [-1, 0, -1, -1, 0], [-1, -1, 0, -1, 0], [-1, -1, -1, 0, 0], [1, 1, 1, 1, 1]]
		self.properties = [
			Property("ExpUtil", PropertyExpression("e_max_s", [0, PropertyExpression("ap", [1])])),
		]
		self.variables = [
			VariableInfo("loc_2", None, "int", 0, 2),
			VariableInfo("t_2", None, "int", 0, 9),
			VariableInfo("boost_1", None, "int", 0, 1),
			VariableInfo("user_1", None, "int", 0, 2),
			VariableInfo("t_1", None, "int", 0, 9),
			VariableInfo("loc_1", None, "int", 0, 2),
			VariableInfo("failure_1", None, "bool"),
			VariableInfo("battery_load", None, "int", 0, 100),
			VariableInfo("Process_2_location", 0, "int", 0, 2),
			VariableInfo("Resources_location", 1, "int", 0, 2),
			VariableInfo("Process_1_location", 2, "int", 0, 2)
		]
		self._aut_Process_2 = Process_2Automaton(self)
		self._aut_Resources = ResourcesAutomaton(self)
		self._aut_Process_1 = Process_1Automaton(self)
		self._aut_Battery = BatteryAutomaton(self)
		self.components = [self._aut_Process_2, self._aut_Resources, self._aut_Process_1, self._aut_Battery]
		self._initial_transient = self._get_initial_transient()
	
	def get_initial_state(self) -> State:
		state = State()
		state.loc_2 = 2
		state.t_2 = 0
		state.boost_1 = 0
		state.user_1 = 0
		state.t_1 = 0
		state.loc_1 = 2
		state.failure_1 = False
		state.battery_load = 100
		self._aut_Process_2.set_initial_values(state)
		self._aut_Resources.set_initial_values(state)
		self._aut_Process_1.set_initial_values(state)
		self._aut_Battery.set_initial_values(state)
		return state
	
	def _get_initial_transient(self) -> Transient:
		transient = Transient()
		transient.label_localFailure = False
		transient.label_noLocalFailure = False
		transient.label_emptyBattery = False
		transient.label_process_1_finishes = False
		transient.label_process_2_finishes = False
		transient.energyLocal = 0
		transient.utilityLocal = 0
		self._aut_Process_2.set_initial_transient_values(transient)
		self._aut_Resources.set_initial_transient_values(transient)
		self._aut_Process_1.set_initial_transient_values(transient)
		self._aut_Battery.set_initial_transient_values(transient)
		return transient
	
	def get_expression_value(self, state: State, expression: int):
		if expression == 0:
			return self.network._get_transient_value(state, "utilityLocal")
		elif expression == 1:
			return (state.battery_load == 0)
		else:
			raise IndexError
	
	def _get_jump_expression_value(self, state: State, transient: Transient, expression: int):
		if expression == 0:
			return transient.utilityLocal
		elif expression == 1:
			return (state.battery_load == 0)
		else:
			raise IndexError
	
	def _get_transient_value(self, state: State, transient_variable: str):
		# Query the automata for the current value of the transient variable
		result = self._aut_Process_2.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_Resources.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_Process_1.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		result = self._aut_Battery.get_transient_value(state, transient_variable)
		if result is not None:
			return result
		# No automaton has a value: return the transient variable's (cached) initial value
		return getattr(self._initial_transient, transient_variable)
	
	def get_transitions(self, state: State) -> List[Transition]:
		# Collect all automaton transitions, gathered by label
		transitions = []
		trans_Process_2 = [[], []]
		transition_count = self._aut_Process_2.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_Process_2.get_guard_value(state, i):
				trans_Process_2[self._aut_Process_2.get_transition_label(state, i)].append(i)
		trans_Resources = [[], []]
		transition_count = self._aut_Resources.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_Resources.get_guard_value(state, i):
				trans_Resources[self._aut_Resources.get_transition_label(state, i)].append(i)
		trans_Process_1 = [[], []]
		transition_count = self._aut_Process_1.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_Process_1.get_guard_value(state, i):
				trans_Process_1[self._aut_Process_1.get_transition_label(state, i)].append(i)
		trans_Battery = [[], []]
		transition_count = self._aut_Battery.get_transition_count(state)
		for i in range(transition_count):
			if self._aut_Battery.get_guard_value(state, i):
				trans_Battery[self._aut_Battery.get_transition_label(state, i)].append(i)
		# Match automaton transitions onto synchronisation vectors
		for svi in range(len(self.sync_vectors)):
			sv = self.sync_vectors[svi]
			synced = [[-1, -1, -1, -1, -1]]
			# Process_2
			if synced is not None:
				if sv[0] != -1:
					if len(trans_Process_2[sv[0]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][0] = trans_Process_2[sv[0]][0]
						for i in range(1, len(trans_Process_2[sv[0]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][0] = trans_Process_2[sv[0]][i]
			# Resources
			if synced is not None:
				if sv[1] != -1:
					if len(trans_Resources[sv[1]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][1] = trans_Resources[sv[1]][0]
						for i in range(1, len(trans_Resources[sv[1]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][1] = trans_Resources[sv[1]][i]
			# Process_1
			if synced is not None:
				if sv[2] != -1:
					if len(trans_Process_1[sv[2]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][2] = trans_Process_1[sv[2]][0]
						for i in range(1, len(trans_Process_1[sv[2]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][2] = trans_Process_1[sv[2]][i]
			# Battery
			if synced is not None:
				if sv[3] != -1:
					if len(trans_Battery[sv[3]]) == 0:
						synced = None
					else:
						existing = len(synced)
						for i in range(existing):
							synced[i][3] = trans_Battery[sv[3]][0]
						for i in range(1, len(trans_Battery[sv[3]])):
							for j in range(existing):
								synced.append(synced[j][:])
								synced[-1][3] = trans_Battery[sv[3]][i]
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
		combs = [[-1, -1, -1, -1]]
		probs = [1.0]
		if transition.transitions[0] != -1:
			existing = len(combs)
			branch_count = self._aut_Process_2.get_branch_count(state, transition.transitions[0])
			for i in range(1, branch_count):
				probability = self._aut_Process_2.get_probability_value(state, transition.transitions[0], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][0] = i
					probs.append(probs[j] * probability)
			probability = self._aut_Process_2.get_probability_value(state, transition.transitions[0], 0)
			for i in range(existing):
				combs[i][0] = 0
				probs[i] *= probability
		if transition.transitions[1] != -1:
			existing = len(combs)
			branch_count = self._aut_Resources.get_branch_count(state, transition.transitions[1])
			for i in range(1, branch_count):
				probability = self._aut_Resources.get_probability_value(state, transition.transitions[1], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][1] = i
					probs.append(probs[j] * probability)
			probability = self._aut_Resources.get_probability_value(state, transition.transitions[1], 0)
			for i in range(existing):
				combs[i][1] = 0
				probs[i] *= probability
		if transition.transitions[2] != -1:
			existing = len(combs)
			branch_count = self._aut_Process_1.get_branch_count(state, transition.transitions[2])
			for i in range(1, branch_count):
				probability = self._aut_Process_1.get_probability_value(state, transition.transitions[2], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][2] = i
					probs.append(probs[j] * probability)
			probability = self._aut_Process_1.get_probability_value(state, transition.transitions[2], 0)
			for i in range(existing):
				combs[i][2] = 0
				probs[i] *= probability
		if transition.transitions[3] != -1:
			existing = len(combs)
			branch_count = self._aut_Battery.get_branch_count(state, transition.transitions[3])
			for i in range(1, branch_count):
				probability = self._aut_Battery.get_probability_value(state, transition.transitions[3], i)
				for j in range(existing):
					combs.append(combs[j][:])
					combs[-1][3] = i
					probs.append(probs[j] * probability)
			probability = self._aut_Battery.get_probability_value(state, transition.transitions[3], 0)
			for i in range(existing):
				combs[i][3] = 0
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
		for i in range(0, 1):
			target_state = State()
			state.copy_to(target_state)
			target_transient = Transient()
			transient.copy_to(target_transient)
			if transition.transitions[0] != -1:
				self._aut_Process_2.jump(state, transient, transition.transitions[0], branch.branches[0], i, target_state, target_transient)
			if transition.transitions[1] != -1:
				self._aut_Resources.jump(state, transient, transition.transitions[1], branch.branches[1], i, target_state, target_transient)
			if transition.transitions[2] != -1:
				self._aut_Process_1.jump(state, transient, transition.transitions[2], branch.branches[2], i, target_state, target_transient)
			if transition.transitions[3] != -1:
				self._aut_Battery.jump(state, transient, transition.transitions[3], branch.branches[3], i, target_state, target_transient)
			state = target_state
			transient = target_transient
		for i in range(len(expressions)):
			expressions[i] = self._get_jump_expression_value(state, transient, expressions[i])
		return state
	
	def jump_np(self, state: State, transition: Transition, expressions: List[int] = []) -> State:
		return self.jump(state, transition, self.get_branches(state, transition)[0], expressions)
