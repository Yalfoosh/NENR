from typing import Tuple

import defaults


class Domain:
    def __init__(self, integer_range: Tuple[int, int] = defaults.DEFAULT_DOMAIN_RANGE):
        if integer_range is None or not isinstance(integer_range, tuple) or not len(integer_range) == 2\
                or not (isinstance(integer_range[0], int) and isinstance(integer_range[1], int))\
                or not (integer_range[0] - integer_range[1] < 0):
            integer_range = defaults.DEFAULT_DOMAIN_RANGE

        self.__members = tuple(range(*integer_range))

    @staticmethod
    def from_domains_merge(domains: Tuple["Domain", "Domain"]):
        if domains is None or not isinstance(domains, tuple) or not len(domains) == 2\
                or not (isinstance(domains[0], Domain) and isinstance(domains[1], Domain)):
            raise ValueError("Wrong domain input.")

        to_return = Domain((0, 1))

        to_return.__members = tuple(sorted(set(domains[0].__members + domains[1].__members)))

        return to_return

    @staticmethod
    def from_domains(domains: Tuple["Domain", "Domain"]):
        if domains is None or not isinstance(domains, tuple) or not len(domains) == 2\
                or not (isinstance(domains[0], Domain) and isinstance(domains[1], Domain)):
            raise ValueError("Wrong domain input.")

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

    def cardinality(self):
        return len(self)

    def index(self, element, start=None, end=None):
        if start is None or not isinstance(start, int):
            start = 0

        if end is None or not isinstance(end, int):
            end = len(self.__members)

        return self.__members.index(element, start, end)

    def __getitem__(self, key):
        return self.__members[key]

    def __len__(self):
        return len(self.__members)

    def __str__(self):
        return str(self.__members)
