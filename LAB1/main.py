from domain import Domain
from fuzzy_set import Fuzzy, MutableFuzzySet, CalculatedFuzzySet
from operation import Operation


def test_1():
    print("Test 1:\n")

    domains = (Domain((0, 5)), Domain((0, 3)))
    merged_domain = Domain.from_domains(domains)

    domains = (*domains, merged_domain)

    for i, domain in enumerate(domains, 1):
        print("Elementi domene D{}: {}.".format(i, domain))
        print("Kardinalitet domene D{} je {}.\n".format(i, domain.cardinality()))

    for index in (0, 5, 14):
        print("Element {} domene D{} je {}.".format(index, 3, domains[2][index]))

    query = (4, 1)
    print("\nIndeks {} u domeni D{} je {}.".format(query, 3, domains[2].index(query)))

    print("\n")


def test_2():
    print("Test 2:\n")

    domains = (Domain((0, 11)), Domain((-5, 6)))
    func = Fuzzy.fuzzy_lambda(domains[1].index(-4), domains[1].index(0), domains[1].index(4))
    sets = (MutableFuzzySet(domains[0]), CalculatedFuzzySet(domains[1], func))

    for i, value in enumerate((1., 0.8, 0.6, 0.4, 0.2)):
        sets[0].set(i, value)

    for s in sets:
        print("{}\n".format(s))

    print("\n")


def test_3():
    print("Test 3:\n")

    fuzzy_set = MutableFuzzySet(Domain((0, 11)))

    for i, value in enumerate((1., 0.8, 0.6, 0.4, 0.2)):
        fuzzy_set.set(i, value)

    not_fuzzy_set = Operation.not_zadeh(fuzzy_set)
    union_set = Operation.or_zadeh(fuzzy_set, not_fuzzy_set)
    hamacher_set = Operation.t_norm_hamacher(fuzzy_set, not_fuzzy_set, 1.)

    sets = (fuzzy_set, not_fuzzy_set, union_set, hamacher_set)
    title = ("Set", "Not set", "Set U Not set", "Hamacher T-Norm of Set and Not Set, p = 1.0")

    for s, t in zip(sets, title):
        print("{}:\n{}\n".format(t, s))


def test():
    test_1()
    test_2()
    test_3()


test()
