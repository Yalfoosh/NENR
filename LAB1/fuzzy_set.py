from typing import Callable, Tuple

import defaults
from domain import Domain


# region Fuzzy Set Implementations
class Fuzzy:
    @property
    def domain(self):
        raise NotImplementedError

    def get_membership(self, domain_element: int or Tuple, start=None, end=None):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    @staticmethod
    def fuzzy_lambda(min_from: int, peak: int, min_to: int):
        if min_from is None or peak is None or min_to is None:
            raise ValueError(defaults.LAMBDA_FUNCTION_NONE_PARAMETERS)

        if not (isinstance(min_from, int) and isinstance(peak, int) and isinstance(min_to, int)):
            raise TypeError(defaults.LAMBDA_FUNCTION_NON_INT_PARAMETERS)

        if not min_from < peak < min_to:
            raise ValueError(defaults.LAMBDA_FUNCTION_WRONG_ORDER_PARAMETERS)

        from_derivative = 1. / (peak - min_from)
        to_derivative = 1. / (min_to - peak)

        return lambda x: 0. if (x <= min_from or x >= min_to)\
                            else 1. if x == peak\
                                    else 1. - (peak - x) * from_derivative if x < peak\
                                                                           else 1 - (x - peak) * to_derivative


class MutableFuzzySet(Fuzzy):
    def __init__(self, domain: Domain):
        if domain is None:
            raise ValueError(defaults.DOMAIN_FUZZY_SET_NONE)

        if not isinstance(domain, Domain):
            raise TypeError

        self.__domain = domain
        self.__membership = [0. for _ in domain]

    @property
    def domain(self):
        return self.__domain

    def set(self, index: int, value: float):
        self.__membership[index] = value

    def get_membership(self, domain_element: int or Tuple, start=None, end=None):
        try:
            return self[self.__domain.index(domain_element, start, end)]
        except ValueError:
            return 0.

    def __getitem__(self, key):
        return self.__membership[key]

    def __str__(self):
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
    def domain(self):
        return self.__domain

    def get_membership(self, domain_element: int or Tuple, start=None, end=None):
        try:
            return self[self.__domain.index(domain_element, start, end)]
        except ValueError:
            return 0.

    def __getitem__(self, key):
        return self.__function(key)

    def __str__(self):
        to_return = ""

        for i, element in enumerate(self.__domain):
            to_return += "μ({}) = {}\n".format(element, round(self[i], 3))

        return to_return[:-1]
# endregion
