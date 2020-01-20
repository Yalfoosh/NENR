import DZ1.defaults
from DZ1.domain import Domain
from DZ1.fuzzy_set import Fuzzy, MutableFuzzySet


class Operation:
    @staticmethod
    def and_zadeh(fuzzy_set_1: Fuzzy, fuzzy_set_2: Fuzzy) -> MutableFuzzySet:
        if fuzzy_set_1 is None or fuzzy_set_2 is None:
            raise ValueError(DZ1.defaults.OPERATION_SET_NONE)

        to_return = MutableFuzzySet(Domain.from_domains_merge((fuzzy_set_1.domain, fuzzy_set_2.domain)))

        for i, element in enumerate(to_return.domain):
            to_return.set(i, min(fuzzy_set_1.get_membership(element), fuzzy_set_2.get_membership(element)))

        return to_return

    @staticmethod
    def not_zadeh(fuzzy_set: Fuzzy) -> MutableFuzzySet:
        if fuzzy_set is None:
            raise ValueError(DZ1.defaults.OPERATION_SET_NONE)

        to_return = MutableFuzzySet(fuzzy_set.domain)

        for i in range(to_return.domain.cardinality()):
            to_return.set(i, 1. - fuzzy_set[i])

        return to_return

    @staticmethod
    def or_zadeh(fuzzy_set_1: Fuzzy, fuzzy_set_2: Fuzzy) -> MutableFuzzySet:
        if fuzzy_set_1 is None or fuzzy_set_2 is None:
            raise ValueError(DZ1.defaults.OPERATION_SET_NONE)

        to_return = MutableFuzzySet(Domain.from_domains_merge((fuzzy_set_1.domain, fuzzy_set_2.domain)))

        for i, element in enumerate(to_return.domain):
            to_return.set(i, max(fuzzy_set_1.get_membership(element), fuzzy_set_2.get_membership(element)))

        return to_return

    @staticmethod
    def s_norm_hamacher(fuzzy_set_1: Fuzzy, fuzzy_set_2: Fuzzy, p: float = 1.0) -> MutableFuzzySet:
        if fuzzy_set_1 is None or fuzzy_set_2 is None:
            raise ValueError(DZ1.defaults.OPERATION_SET_NONE)

        if p is None or not isinstance(p, float):
            raise TypeError(DZ1.defaults.OPERATION_HAMACHER_T_NORM_WRONG_TYPE)

        if p < 0.:
            raise ValueError(DZ1.defaults.OPERATION_HAMACHER_NORM_NEGATIVE_PARAMETER)

        to_return = MutableFuzzySet(Domain.from_domains_merge((fuzzy_set_1.domain, fuzzy_set_2.domain)))

        for i, element in enumerate(to_return.domain):
            a, b = (fuzzy_set_1.get_membership(element), fuzzy_set_2.get_membership(element))

            if a == b and a == 0.:
                to_return.set(i, 0.)
            else:
                to_return.set(i, (a + b - (2 - p) * a * b) / (1 - (1 - p) * a * b))

        return to_return

    @staticmethod
    def t_norm_hamacher(fuzzy_set_1: Fuzzy, fuzzy_set_2: Fuzzy, p: float = 1.) -> MutableFuzzySet:
        if fuzzy_set_1 is None or fuzzy_set_2 is None:
            raise ValueError(DZ1.defaults.OPERATION_SET_NONE)

        if p is None or not isinstance(p, float):
            raise TypeError(DZ1.defaults.OPERATION_HAMACHER_T_NORM_WRONG_TYPE)

        if p < 0.:
            raise ValueError(DZ1.defaults.OPERATION_HAMACHER_NORM_NEGATIVE_PARAMETER)

        to_return = MutableFuzzySet(Domain.from_domains_merge((fuzzy_set_1.domain, fuzzy_set_2.domain)))

        for i, element in enumerate(to_return.domain):
            a, b = (fuzzy_set_1.get_membership(element), fuzzy_set_2.get_membership(element))

            if a == b and a == 0.:
                to_return.set(i, 0.)
            else:
                to_return.set(i, (a * b) / (p + (1 - p) * (a + b + a * b)))

        return to_return
