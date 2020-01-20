from typing import Callable, Tuple

import DZ1.defaults as defaults
from DZ1.domain import Domain


# region Fuzzy Set Implementations
class Fuzzy:
    @property
    def domain(self) -> Domain:
        raise NotImplementedError

    def get_membership(self, domain_element: int or Tuple, start=None, end=None):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    @staticmethod
    def fuzzy_gamma(alpha: int, beta: int):
        if alpha is None or beta is None:
            raise ValueError(defaults.GAMMA_FUNCTION_NONE_PARAMETERS)

        if not (isinstance(alpha, int) and isinstance(beta, int)):
            raise TypeError(defaults.GAMMA_FUNCTION_NON_INT_PARAMETERS)

        if not alpha < beta:
            raise ValueError(defaults.GAMMA_FUNCTION_WRONG_ORDER_PARAMETERS)

        return lambda x: 0. if x <= alpha \
            else 1. if x >= beta \
            else (x - alpha) / (beta - alpha)

    @staticmethod
    def fuzzy_lambda(alpha: int, beta: int, gamma: int):
        if alpha is None or beta is None or gamma is None:
            raise ValueError(defaults.LAMBDA_FUNCTION_NONE_PARAMETERS)

        if not (isinstance(alpha, int) and isinstance(beta, int) and isinstance(gamma, int)):
            raise TypeError(defaults.LAMBDA_FUNCTION_NON_INT_PARAMETERS)

        if not alpha < beta < gamma:
            raise ValueError(defaults.LAMBDA_FUNCTION_WRONG_ORDER_PARAMETERS)

        from_derivative = beta - alpha
        to_derivative = gamma - beta

        return lambda x: 0. if (x <= alpha or x >= gamma)\
                            else 1. if x == beta\
                                    else 1. - (beta - x) / from_derivative if x < beta\
                                                                           else 1 - (x - beta) / to_derivative

    @staticmethod
    def fuzzy_pi(alpha: int, beta: int, gamma: int, delta: int):
        if alpha is None or beta is None or gamma is None or delta is None:
            raise ValueError(defaults.PI_FUNCTION_NONE_PARAMETERS)

        if not (isinstance(alpha, int) and isinstance(beta, int) and isinstance(gamma, int) and isinstance(delta, int)):
            raise TypeError(defaults.PI_FUNCTION_NON_INT_PARAMETERS)

        if not alpha < beta < gamma < delta:
            raise ValueError(defaults.PI_FUNCTION_WRONG_ORDER_PARAMETERS)

        from_derivative = beta - alpha
        to_derivative = delta - gamma

        return lambda x: 0. if (x <= alpha or x >= delta)\
                            else 1. if beta <= x <= gamma\
                                    else (x - alpha) / from_derivative if x < beta\
                                                                       else (delta - x) / to_derivative


# region Specific Implementations
class MutableFuzzySet(Fuzzy):
    def __init__(self, domain: Domain):
        if domain is None:
            raise ValueError(defaults.DOMAIN_FUZZY_SET_NONE)

        if not isinstance(domain, Domain):
            raise TypeError

        self.__domain = domain
        self.__membership = [0. for _ in domain]

    @property
    def domain(self) -> Domain:
        return self.__domain

    def set(self, index: int, value: float):
        self.__membership[index] = value

    def get_membership(self, domain_element: int or Tuple, start=None, end=None) -> float:
        try:
            return self[self.__domain.index(domain_element, start, end)]
        except ValueError:
            return 0.

    def __getitem__(self, key) -> float:
        return self.__membership[key]

    def __str__(self) -> str:
        to_return = ""

        for i, element in enumerate(self.__domain):
            to_return += "μ({}) = {}\n".format(element, round(self[i], 3))

        return to_return[:-1]


class CalculatedFuzzySet(Fuzzy):
    def __init__(self, domain: Domain, function: Callable):
        if domain is None:
            raise ValueError(defaults.DOMAIN_FUZZY_SET_NONE)

        if not isinstance(domain, Domain):
            raise TypeError

        if function is None or not callable(function):
            self.__function = lambda x: 0.
        else:
            self.__function = function

        self.__domain = domain

    @property
    def domain(self) -> Domain:
        return self.__domain

    def get_membership(self, domain_element: int or Tuple, start=None, end=None) -> float:
        try:
            return self[self.__domain.index(domain_element, start, end)]
        except ValueError:
            return 0.

    def __getitem__(self, key) -> float:
        return self.__function(key)

    def __str__(self) -> str:
        to_return = ""

        for i, element in enumerate(self.__domain):
            to_return += "μ({}) = {}\n".format(element, round(self[i], 3))

        return to_return[:-1]
# endregion
# endregion
