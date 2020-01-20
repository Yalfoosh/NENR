from copy import deepcopy
import numpy as np
import re
from typing import Callable, List


dataset_separator = re.compile(r"\s+")


# region Parsing
def parse(file_path: str):
    to_return = list()

    with open(file_path, mode="r") as file:
        for line in file.readlines():
            if line is not None and len(line) != 0:
                results = tuple(map(float, map(str.strip, dataset_separator.split(line, maxsplit=2))))
                to_return.append((tuple(results[:2]), results[2]))

    return to_return
# endregion


# region Task Specific
def the_function(x, y, chromosome):
    a, b, c, d, e = chromosome

    term_1 = np.sin(np.add(a, np.multiply(b, x)))
    term_2_first_factor = np.multiply(c, np.cos(np.multiply(x, np.add(d, y))))
    term_2_second_factor = np.reciprocal(np.add(1, np.power(np.e, np.square(np.subtract(x, e)))))

    return np.add(term_1, np.multiply(term_2_first_factor, term_2_second_factor))


def square_error(y: float, y_pred: float):
    return np.square(np.subtract(y, y_pred))


def mse(y: List[float], y_pred: List[float]):
    return np.mean([square_error(y[i], y_pred[i]) for i in range(len(y))])


def chromosome_loss(x: List, y_pred: List, chromosome: np.ndarray,
                    loss_function: Callable = mse, prediction_function: Callable = the_function):
    return loss_function([prediction_function(*x_and_y, chromosome) for x_and_y in x], y_pred)
# endregion


# region Helper Classes
class Roulette:
    def __init__(self, members_with_scores: List[List]):
        self.members_with_scores = sorted(members_with_scores, key=lambda x: x[1], reverse=True)
        total_score = np.sum(list(map(lambda x: x[1], self.members_with_scores)))

        self.members_with_scores[0][1] /= total_score

        for i in range(1, len(self.members_with_scores)):
            self.members_with_scores[i][1] /= total_score
            self.members_with_scores[i][1] += self.members_with_scores[i - 1][1]

    def run(self, times: int = 1):
        if times >= len(self.members_with_scores):
            return list(map(lambda x: x[0], self.members_with_scores))

        members = deepcopy(self.members_with_scores)
        to_return = list()

        for _ in range(times):
            result = np.random.rand()

            for i in range(len(members)):
                if result < members[i][1]:
                    to_return.append(members[i][0])
                    del members[i]
                    break

            to_return.append(members[-1][0])
            del members[-1]

        return to_return
# endregion


# region Scaling Functions
def constant(initial_value):
    def _y(**_):
        return initial_value

    return _y


def natural_iteration_decay(initial_value, half_life):
    def _y(**kwargs):
        iteration = kwargs.get("iteration", 0)

        return np.power(0.5, np.divide(iteration, half_life)) * initial_value

    return _y


def linear_iteration_decay(initial_value, differential, upper_bound=None, lower_bound=None):
    def _y(**kwargs):
        iteration = kwargs.get("iteration", 0)

        to_return = initial_value - iteration * differential

        if upper_bound is not None and to_return > upper_bound:
            return upper_bound

        if lower_bound is not None and to_return < lower_bound:
            return lower_bound

        return to_return

    return _y


def explicit(default_value, value_map):
    def _y(**kwargs):
        iteration = kwargs.get("iteration", 0)

        return value_map.get(iteration, default_value)

    return _y


def tanh_log_iteration(initial_value, final_value):
    differential = final_value - initial_value

    def _y(**kwargs):
        iteration = kwargs.get("iteration", 0)

        return initial_value + np.tanh(np.log10(max(iteration, 1))) * differential

    return _y


def lambda_scale(initial_value):
    def _y(**kwargs):
        scale_function = kwargs.get("scale_function", lambda x: initial_value)(initial_value)

        return scale_function(kwargs)

    return _y
# endregion


# region Selection Functions
def generative_select(**kwargs):
    population = kwargs.get("population", [])
    max_population = kwargs.get("max_population", 0)
    wellness_function = kwargs.get("wellness_function", lambda x: 1)

    elitism = min(kwargs.get("elitism", 0), len(population))
    parent_count = kwargs.get("parent_count", 2)

    new_population = list()
    to_cross = list()

    pop_with_scores = sorted(list(map(lambda x: [x, wellness_function(x)], population)), key=lambda x: -x[1])

    for el in range(elitism):
        new_population.append(pop_with_scores[el][0])

    roulette = Roulette(pop_with_scores)
    number_of_crosses = max_population - len(new_population)

    for _ in range(number_of_crosses):
        parents = roulette.run(parent_count)

        to_cross.append(parents)

    return new_population, to_cross


def three_tournament_select(**kwargs):
    population = kwargs.get("population", [])
    mortality = kwargs.get("mortality", constant(1))

    wellness_function = kwargs.get("wellness_function", lambda x: 1)

    to_cross = list()

    mortality = max(1, int(mortality(**kwargs)))
    mortality = min(mortality, len(population) - 2)
    for i in range(mortality):
        choices = np.random.choice(len(population), 3)

        lowest_id = 0

        for j in range(1, 3):
            if wellness_function(population[choices[j]]) < wellness_function(population[choices[lowest_id]]):
                lowest_id = j

        parents = list()

        for j in range(0, 3):
            if j != lowest_id:
                parents.append(population[choices[j]])

        to_cross.append(parents)
        del population[choices[lowest_id]]

    return population, to_cross
# endregion


# region Cross Functions
def stochastic_cross(parents: List[np.ndarray], **kwargs):
    mommy, daddy = parents[:2]
    mommy_on_top = kwargs.get("priority_function", lambda: np.random.rand() < 0.5)

    baby = list()

    for i in range(len(mommy)):
        baby.append(mommy[i] if mommy_on_top() else daddy[i])

    return np.array(baby)


def mean_cross(parents: List[np.ndarray], **_):
    return np.mean(parents, axis=0)


def double_cross(parents: List[np.ndarray], **kwargs):
    iteration = kwargs.get("iteration", 0)

    if iteration < kwargs.get("change_iteration", 100):
        return stochastic_cross(parents, **kwargs)
    else:
        return mean_cross(parents, **kwargs)
# endregion


# region Mutation Functions
def stochastic_swap_mutate(candidate: np.ndarray, **kwargs):
    probability = kwargs.get("swap_probability", constant(0.01))

    candidate_length = len(candidate)

    if np.random.rand() < probability(**kwargs):
        first = np.random.randint(0, candidate_length)
        second = (first + np.random.randint(1, candidate_length)) % candidate_length

        candidate[first], candidate[second] = candidate[second], candidate[first]

    return candidate


def gaussian_mutate(candidate: np.ndarray, **kwargs):
    probability = kwargs.get("gauss_probability", constant(0.01))
    intensity = kwargs.get("gauss_intensity", constant(1))

    for i in range(len(candidate)):
        if np.random.rand() < probability(**kwargs):
            candidate[i] = np.random.normal(candidate[i], intensity(**kwargs))

    return candidate


def normalization_mutate(candidate: np.ndarray, **kwargs):
    probability = kwargs.get("normalization_probability", constant(0.05))
    intensity = kwargs.get("normalization_intensity", constant(10))
    gene_interval = kwargs.get("gene_interval", (0., 1.))

    if np.random.rand() < probability(**kwargs):
        for i in range(len(candidate)):
            if candidate[i] > gene_interval[1]:
                candidate[i] = np.divide(candidate[i], intensity(**kwargs))
            elif candidate[i] < gene_interval[0]:
                candidate[i] = np.multiply(candidate[i], intensity(**kwargs))

    return candidate
# endregion
