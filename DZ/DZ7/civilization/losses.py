from typing import Callable, Tuple

import numpy as np

from civilization.display_strategies import DisplayStrategy


class Loss:
    def __init__(self, display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        self.__call_count = 0
        self.__display_strategy = display_strategy
        self.__change_display_before_call = change_display_before_call

    @property
    def call_count(self):
        return self.__call_count

    @property
    def display_strategy(self):
        return self.__display_strategy

    def reset(self):
        self.__call_count = 0

    def call(self, value: np.ndarray, **kwargs):
        self.__call_count += 1

        return self._call(value if not self.__change_display_before_call
                                else self.__display_strategy.decode(value, **kwargs), **kwargs)

    def _call(self, value: np.ndarray, **kwargs):
        raise NotImplementedError

    def fitness(self, value: np.ndarray, **kwargs):
        loss_value = kwargs.get("loss_value", None)

        if loss_value is None:
            loss_value = self.call(value)

        return -loss_value

    def logits(self, fitnesses, **kwargs):
        fitness_exps = np.exp(np.asarray(fitnesses))
        fitness_exp_sum = np.sum(fitness_exps)

        return [fitness_exp / fitness_exp_sum for fitness_exp in fitness_exps]


class LambdaLoss(Loss):
    def __init__(self, function: Callable, display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        super().__init__(display_strategy, change_display_before_call)

        self.__function = function

    def _call(self, value: np.ndarray, **kwargs):
        return self.__function(value, **kwargs)


class RosenbrockBananaLoss(Loss):
    def __init__(self, display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        super().__init__(display_strategy, change_display_before_call)

    def _call(self, value: np.ndarray, **kwargs):
        return 100 * np.square(value[1] - value[0] ** 2) + np.square(1. - value[0])


class SumSquareError(Loss):
    def __init__(self, display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        super().__init__(display_strategy, change_display_before_call)

    def _call(self, value: np.ndarray, **kwargs):
        return np.sum(np.square(value - np.arange(1, len(value) + 1)))


class SchafferLoss(Loss):
    def __init__(self, display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        super().__init__(display_strategy, change_display_before_call)

    def _call(self, value: np.ndarray, **kwargs):
        square_sum = np.sum(np.square(value))

        return 0.5 + np.divide(np.square(np.sin(np.sqrt(square_sum))) - 0.5,
                               np.square(1. + 1e-3 * square_sum))


class SchafferApproximateLoss(Loss):
    def __init__(self, display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        super().__init__(display_strategy, change_display_before_call)

    def _call(self, value: np.ndarray, **kwargs):
        square_sum = np.sum(np.square(value))
        ss_1 = np.power(square_sum, 0.25)
        ss_2 = np.power(square_sum, 0.1)

        return ss_1 * (1. + np.square(np.sin(50 * ss_2)))


class DatasetNNMeanSquareError(Loss):
    def __init__(self, model, validation_data: Tuple[np.ndarray, np.ndarray],
                 display_strategy: DisplayStrategy, change_display_before_call: bool = True):
        super().__init__(display_strategy, change_display_before_call)

        self.__model = model
        self.__val_x = validation_data[0]
        self.__val_y = validation_data[1]

    def _call(self, value: np.ndarray, **kwargs):
        self.__model.inject_parameters(value)

        _results = np.array([self.__model.predict(x) for x in self.__val_x])
        _diff = np.square(self.__val_y - _results)

        return np.sum(_diff) / len(self.__val_y)
