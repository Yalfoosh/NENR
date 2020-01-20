import defaults

from DZ1.domain import Domain
from DZ1.fuzzy_set import Fuzzy, MutableFuzzySet


class Relations:
    @staticmethod
    def is_UxU_relation(relation: Fuzzy) -> bool:
        return all(filter(lambda x: len(x) == 2, relation.domain))

    @staticmethod
    def is_symmetric(relation: Fuzzy) -> bool:
        if Relations.is_UxU_relation(relation):
            for element in relation.domain:
                if relation.get_membership(element) != relation.get_membership(element[::-1]):
                    return False

            return True

        return False

    @staticmethod
    def is_reflexive(relation: Fuzzy) -> bool:
        if Relations.is_UxU_relation(relation):
            for element in relation.domain:
                if element[0] == element[1] and relation.get_membership(element) != 1.:
                    return False

            return True

        return False

    @staticmethod
    def is_max_min_transitive(relation: Fuzzy) -> bool:
        if Relations.is_UxU_relation(relation):
            for i, element_i in enumerate(relation.domain):
                for element_j in relation.domain[i + 1:]:
                    if element_i[1] == element_j[0]:
                        new_element = (element_i[0], element_j[1])

                        if relation.get_membership(new_element) < min([relation.get_membership(element_i),
                                                                       relation.get_membership(element_j)]):
                            return False

            return True

        return False

    @staticmethod
    def binary_relation_composite(first: Fuzzy, second: Fuzzy) -> MutableFuzzySet:
        if not (Relations.is_UxU_relation(first) and Relations.is_UxU_relation(second)):
            raise ValueError(defaults.RELATIONS_ARE_NOT_BINARY.format(first, second))

        first_components, second_components = [x.domain.decompose() for x in (first, second)]

        if first_components[-1] != second_components[0]:
            raise ValueError(defaults.RELATIONS_ARE_NOT_CHAINED)

        U = Domain(first_components[0])
        V = Domain(first_components[-1])
        W = Domain(second_components[-1])

        to_return = MutableFuzzySet(Domain.from_domains((U, W)))

        for element_i in U:
            for element_k in W:
                max_value = 0

                for element_j in V:
                    max_value = max(max_value,
                                    min(first.get_membership((element_i, element_j)),
                                        second.get_membership((element_j, element_k))))

                to_return.set(to_return.domain.index((element_i, element_k)), max_value)

        return to_return

    @staticmethod
    def is_fuzzy_equivalence(relation: Fuzzy) -> bool:
        return Relations.is_symmetric(relation)\
               and Relations.is_reflexive(relation)\
               and Relations.is_max_min_transitive(relation)
