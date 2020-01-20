from typing import Callable, List

from DZ1.domain import Domain
from DZ1.fuzzy_set import Fuzzy, CalculatedFuzzySet, MutableFuzzySet
from DZ1.operation import Operation

# region Variable Domains
max_acceleration = 35
max_map_size = (700, 1000)
wind_velocity = (20, 50)
angle_range = (-90, 90)
ship_radius = 10

max_distance = int((max_map_size[0] ** 2 + max_map_size[1] ** 2) ** 0.5 + 0.5)
max_velocity = int((2 * max_acceleration * max_distance) ** 0.5 + max(wind_velocity) + 0.5)

orientation_domain = Domain((0, 2))
distance_domain = Domain((0, max_distance + 1))
acceleration_domain = Domain((-max_acceleration, max_acceleration + 1))
velocity_domain = Domain((0, max_velocity + 1))
angle_domain = Domain((angle_range[0], angle_range[1] + 1))

wall_thresholds = (max_acceleration,
                   max_acceleration + 1 * ship_radius,
                   max_acceleration + 2 * ship_radius)

velocity_thresholds = (50, 80, 90)
acceleration_thirds = (0,
                       max_acceleration // 3,
                       max_acceleration * 2 // 3,
                       max_acceleration)
angle_thresholds = (50, 70, 85, 90)

wrong_way_bounds = (0, 1)

really_close_to_wall_bounds = (wall_thresholds[0], wall_thresholds[1])
close_to_wall_bounds = (wall_thresholds[1], wall_thresholds[2])

moving_bounds = (0, 10)
going_too_slowly_bounds = (velocity_thresholds[0], velocity_thresholds[1])
going_too_quickly_bounds = (velocity_thresholds[1], velocity_thresholds[2])

slow_down_bounds = (-acceleration_thirds[3], -acceleration_thirds[2])
speed_up_bounds = (acceleration_thirds[2], acceleration_thirds[3])
slow_down_bounds = (max_acceleration + slow_down_bounds[0], max_acceleration + slow_down_bounds[1])
speed_up_bounds = (max_acceleration + speed_up_bounds[0], max_acceleration + speed_up_bounds[1])

turn_left_sharply_bounds = (angle_thresholds[2], angle_thresholds[3])
turn_right_sharply_bounds = (-angle_thresholds[3], -angle_thresholds[2])
turn_around_bounds = (-angle_range[0] + 89, -angle_range[0] + 90)
turn_left_sharply_bounds = (-angle_range[0] + turn_left_sharply_bounds[0],
                            -angle_range[0] + turn_left_sharply_bounds[1])
turn_right_sharply_bounds = (-angle_range[0] + turn_right_sharply_bounds[0],
                             -angle_range[0] + turn_right_sharply_bounds[1])
# endregion


# region Rules
# region Premise Database
wrong_way = CalculatedFuzzySet(orientation_domain, Fuzzy.fuzzy_l(*wrong_way_bounds))

close_to_wall = CalculatedFuzzySet(distance_domain, Fuzzy.fuzzy_l(*close_to_wall_bounds))
really_close_to_wall = CalculatedFuzzySet(distance_domain, Fuzzy.fuzzy_l(*really_close_to_wall_bounds))

going_too_slowly = CalculatedFuzzySet(velocity_domain, Fuzzy.fuzzy_l(*going_too_slowly_bounds))
going_too_quickly = CalculatedFuzzySet(velocity_domain, Fuzzy.fuzzy_gamma(*going_too_quickly_bounds))
# endregion

# region Conclusion Database
slow_down = CalculatedFuzzySet(acceleration_domain, Fuzzy.fuzzy_l(*slow_down_bounds))
speed_up = CalculatedFuzzySet(acceleration_domain, Fuzzy.fuzzy_gamma(*speed_up_bounds))

turn_left_sharply = CalculatedFuzzySet(angle_domain, Fuzzy.fuzzy_gamma(*turn_left_sharply_bounds))
turn_right_sharply = CalculatedFuzzySet(angle_domain, Fuzzy.fuzzy_l(*turn_right_sharply_bounds))
turn_around = CalculatedFuzzySet(angle_domain, Fuzzy.fuzzy_gamma(*turn_around_bounds))
# endregion


def minimum_machine(current_value, current_premise: Fuzzy, element):
    return min(current_value, current_premise.get_membership(element))


def product_machine(current_value, current_premise: Fuzzy, element):
    return current_value * current_premise.get_membership(element)


class Rule:
    def __init__(self, premises: List[Fuzzy], conclusion: Fuzzy):
        self.__premises = premises if premises is not None else list()
        self.__conclusion = conclusion

    def conclude(self, arguments: List, conclusion_function: Callable = minimum_machine):
        membership = 1.

        for i, premise in enumerate(self.__premises):
            if premise is not None:
                membership = conclusion_function(membership, premise, arguments[i])

        new_fuzzy_set = MutableFuzzySet(domain=self.__conclusion.domain)

        for i in range(len(new_fuzzy_set.domain)):
            new_fuzzy_set.set(i, membership * self.__conclusion[i])

        return new_fuzzy_set
# endregion


# region Defuzzifiers
class Defuzzifier:
    def defuzzify(self, fuzzy_set: Fuzzy):
        raise NotImplementedError


class COA(Defuzzifier):
    def defuzzify(self, fuzzy_set: Fuzzy) -> int:
        numerator = 0.
        denominator = 0.

        for domain_element in fuzzy_set.domain:
            t = fuzzy_set.get_membership(domain_element)

            numerator += t * domain_element
            denominator += t

        if abs(numerator) < 10e-6 or abs(denominator) < 10e-6:
            return 0

        return int(numerator / denominator + 0.5)
# endregion


# region Fuzzy Systems

class FuzzySystem:
    def __init__(self, defuzzifier: Defuzzifier, initial_rules: List[Rule] = None):
        self.__rules: List[Rule] = list()

        if initial_rules is not None:
            for initial_rule in initial_rules:
                self.__rules.append(initial_rule)

        self.__defuzzifier: Defuzzifier = defuzzifier

    @property
    def defuzzifier(self):
        return self.__defuzzifier

    def add_rule(self, premises: List[Fuzzy], conclusion: Fuzzy):
        self.__rules.append(Rule(premises, conclusion))

    def rule_count(self):
        return len(self.__rules)

    def get_rule(self, key: int):
        return self.__rules[key]

    def infer(self, values: List, conclusion_function: Callable = minimum_machine):
        if len(self.__rules) == 0:
            return None

        to_return = self.__rules[0].conclude(values, conclusion_function)

        if len(self.__rules) > 1:
            for rule in self.__rules[1:]:
                to_return = Operation.or_zadeh(to_return, rule.conclude(values, conclusion_function))

        return to_return

    def conclude(self, values, conclusion_function: Callable = minimum_machine) -> int:
        t = self.infer(values, conclusion_function)
        return self.__defuzzifier.defuzzify(t) if t is not None else 0


class AccelerationSystem(FuzzySystem):
    def __init__(self, defuzzifier: Defuzzifier, initial_rules: List[Rule] = None):
        super().__init__(defuzzifier, initial_rules)

        self.add_rule([None, None, None, None, going_too_slowly, None], speed_up)
        self.add_rule([None, None, None, None, going_too_quickly, None], slow_down)


class RudderSystem(FuzzySystem):
    def __init__(self, defuzzifier: Defuzzifier, initial_rules: List[Rule] = None):
        super().__init__(defuzzifier, initial_rules)

        self.add_rule([None, None, close_to_wall, None, None, None], turn_right_sharply)
        self.add_rule([None, None, None, close_to_wall, None, None], turn_left_sharply)
        self.add_rule([None, None, None, None, None, wrong_way], turn_around)


# endregion
