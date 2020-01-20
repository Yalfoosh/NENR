from typing import List, Tuple

import DZ1.defaults as defaults


class Domain:
    def __init__(self, integer_range: Tuple[int, int] = defaults.DEFAULT_DOMAIN_RANGE):
        if integer_range is None or not isinstance(integer_range, tuple) or not len(integer_range) == 2\
                or not (isinstance(integer_range[0], int) and isinstance(integer_range[1], int))\
                or not (integer_range[0] - integer_range[1] < 0):
            integer_range = defaults.DEFAULT_DOMAIN_RANGE

        self.__members = tuple(range(*integer_range))

    @staticmethod
    def from_domains(domains: Tuple["Domain", "Domain"] or List["Domain"]) -> "Domain":
        if domains is None or not isinstance(domains, tuple) or not len(domains) == 2\
                or not (isinstance(domains[0], Domain) and isinstance(domains[1], Domain)):
            raise ValueError(defaults.DOMAIN_INPUT_IS_POOPOO)

        members = list()

        for domain_predecessor in domains[0]:
            if isinstance(domain_predecessor, int):
                domain_predecessor = (domain_predecessor, )

            for domain_successor in domains[1]:
                if isinstance(domain_successor, int):
                    domain_successor = (domain_successor, )

                members.append(domain_predecessor + domain_successor)

        to_return = Domain((0, 1))

        to_return.__members = tuple(members)
        return to_return

    @staticmethod
    def from_domains_merge(domains: Tuple["Domain", "Domain"]) -> "Domain":
        if domains is None or not isinstance(domains, tuple) or not len(domains) == 2 \
                or not (isinstance(domains[0], Domain) and isinstance(domains[1], Domain)):
            raise ValueError(defaults.DOMAIN_INPUT_IS_POOPOO)

        to_return = Domain((0, 1))

        to_return.__members = tuple(sorted(set(domains[0].__members + domains[1].__members)))

        return to_return

    def cardinality(self) -> int:
        return len(self)

    def index(self, element, start=None, end=None) -> int:
        if start is None or not isinstance(start, int):
            start = 0

        if end is None or not isinstance(end, int):
            end = len(self.__members)

        return self.__members.index(element, start, end)

    def decompose(self):
        ranges = list()
        max_value = 0

        max_value = max([max(x) for x in self])

        for element in self:
            for i, value in enumerate(element):
                if len(ranges) <= i:
                    ranges.append([max_value, 0])

                ranges[i][0] = min(ranges[i][0], value)
                ranges[i][1] = max(ranges[i][1], value)

        return tuple(map(lambda x: tuple((x[0], x[1] + 1)), ranges))

    def __getitem__(self, key) -> Tuple or int:
        return self.__members[key]

    def __len__(self) -> int:
        return len(self.__members)

    def __str__(self) -> str:
        return str(self.__members)
