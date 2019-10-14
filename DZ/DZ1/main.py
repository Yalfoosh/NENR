from domain import Domain
from fuzzy_set import Fuzzy, MutableFuzzySet, CalculatedFuzzySet
from operation import Operation


# region Tasks
def task_1():
    print("Zadatak [1]:\n")

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


def task_2():
    print("Zadatak [2]:\n")

    domains = (Domain((0, 11)), Domain((-5, 6)))
    func = Fuzzy.fuzzy_lambda(domains[1].index(-4), domains[1].index(0), domains[1].index(4))
    sets = (MutableFuzzySet(domains[0]), CalculatedFuzzySet(domains[1], func))
    titles = ("S₁", "S₂ (lambda<-4, 0, 4>(D₂))")

    for i, value in enumerate((1., 0.8, 0.6, 0.4, 0.2)):
        sets[0].set(i, value)

    for t, s in zip(titles, sets):
        print("{}:\n{}\n".format(t, s))

    print("\n")


def task_3():
    print("Zadatak [3]:\n")

    fuzzy_set = MutableFuzzySet(Domain((0, 11)))

    for i, value in enumerate((1., 0.8, 0.6, 0.4, 0.2)):
        fuzzy_set.set(i, value)

    not_fuzzy_set = Operation.not_zadeh(fuzzy_set)
    union_set = Operation.or_zadeh(fuzzy_set, not_fuzzy_set)
    hamacher_set = Operation.t_norm_hamacher(fuzzy_set, not_fuzzy_set, 1.)

    sets = (fuzzy_set, not_fuzzy_set, union_set, hamacher_set)
    titles = ("S₁", "S₂ (NOT(S₁))", "S₁ ∪ S₂", "Hamacher T-norma <1.0> (S₁, S₂)")

    for t, s in zip(titles, sets):
        print("{}:\n{}\n".format(t, s))


def do_tasks():
    task_1()
    task_2()
    task_3()
# endregion


do_tasks()
