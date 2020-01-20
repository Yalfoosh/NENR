from relations import *

from copy import deepcopy


def conclude(condition):
    return "je" if condition else "nije"


def task_1():
    print("Zadatak [1]\n")

    some_domain = Domain((1, 6))
    composite_domain = Domain.from_domains(tuple(2*[some_domain]))

    r1 = MutableFuzzySet(composite_domain)
    r1.set(composite_domain.index((1, 1)), 1)
    r1.set(composite_domain.index((2, 2)), 1)
    r1.set(composite_domain.index((3, 3)), 1)
    r1.set(composite_domain.index((4, 4)), 1)
    r1.set(composite_domain.index((5, 5)), 1)
    r1.set(composite_domain.index((3, 1)), 0.5)
    r1.set(composite_domain.index((1, 3)), 0.5)

    r2 = MutableFuzzySet(composite_domain)
    r2.set(composite_domain.index((1, 1)), 1)
    r2.set(composite_domain.index((2, 2)), 1)
    r2.set(composite_domain.index((3, 3)), 1)
    r2.set(composite_domain.index((4, 4)), 1)
    r2.set(composite_domain.index((5, 5)), 1)
    r2.set(composite_domain.index((3, 1)), 0.5)
    r2.set(composite_domain.index((1, 3)), 0.1)

    r3 = MutableFuzzySet(composite_domain)
    r3.set(composite_domain.index((1, 1)), 1)
    r3.set(composite_domain.index((2, 2)), 1)
    r3.set(composite_domain.index((3, 3)), 0.3)
    r3.set(composite_domain.index((4, 4)), 1)
    r3.set(composite_domain.index((5, 5)), 1)
    r3.set(composite_domain.index((1, 2)), 0.6)
    r3.set(composite_domain.index((2, 1)), 0.6)
    r3.set(composite_domain.index((2, 3)), 0.7)
    r3.set(composite_domain.index((3, 2)), 0.7)
    r3.set(composite_domain.index((1, 3)), 0.5)
    r3.set(composite_domain.index((3, 1)), 0.5)

    r4 = MutableFuzzySet(composite_domain)
    r4.set(composite_domain.index((1, 1)), 1)
    r4.set(composite_domain.index((2, 2)), 1)
    r4.set(composite_domain.index((3, 3)), 1)
    r4.set(composite_domain.index((4, 4)), 1)
    r4.set(composite_domain.index((5, 5)), 1)
    r4.set(composite_domain.index((1, 2)), 0.4)
    r4.set(composite_domain.index((2, 1)), 0.4)
    r4.set(composite_domain.index((2, 3)), 0.5)
    r4.set(composite_domain.index((3, 2)), 0.5)
    r4.set(composite_domain.index((1, 3)), 0.4)
    r4.set(composite_domain.index((3, 1)), 0.4)

    print("Prva relacija {} definirana nad U x U.".format(conclude(Relations.is_symmetric(r1))))
    print("Prva relacija {} simetrična.".format(conclude(Relations.is_symmetric(r1))))
    print("Druga relacija {} simetrična.".format(conclude(Relations.is_symmetric(r2))))
    print("Prva relacija {} refleksivna.".format(conclude(Relations.is_reflexive(r1))))
    print("Treća relacija {} refleksivna.".format(conclude(Relations.is_reflexive(r3))))
    print("Treća relacija {} max-min tranzitivna.".format(conclude(Relations.is_max_min_transitive(r3))))
    print("Četvrta relacija {} max-min tranzitivna.".format(conclude(Relations.is_max_min_transitive(r4))))


def task_2():
    print("\nZadatak [2]\n")

    U = Domain((1, 5))
    V = Domain((1, 4))
    W = Domain((1, 5))

    UV = Domain.from_domains((U, V))
    VW = Domain.from_domains((V, W))

    r1 = MutableFuzzySet(UV)
    r1.set(UV.index((1, 1)), 0.3)
    r1.set(UV.index((1, 2)), 1)
    r1.set(UV.index((3, 3)), 0.5)
    r1.set(UV.index((4, 3)), 0.5)

    r2 = MutableFuzzySet(VW)
    r2.set(VW.index((1, 1)), 1)
    r2.set(VW.index((2, 1)), 0.5)
    r2.set(VW.index((2, 2)), 0.7)
    r2.set(VW.index((3, 3)), 1)
    r2.set(VW.index((3, 4)), 0.4)

    r1_r2 = Relations.binary_relation_composite(r1, r2)

    print(r1_r2)


def task_3():
    print("\nZadatak [3]\n")

    some_domain = Domain((1, 5))
    composite_domain = Domain.from_domains(tuple(2*[some_domain]))

    r1 = MutableFuzzySet(composite_domain)
    r1.set(composite_domain.index((1, 1)), 1)
    r1.set(composite_domain.index((2, 2)), 1)
    r1.set(composite_domain.index((3, 3)), 1)
    r1.set(composite_domain.index((4, 4)), 1)
    r1.set(composite_domain.index((1, 2)), 0.3)
    r1.set(composite_domain.index((2, 1)), 0.3)
    r1.set(composite_domain.index((2, 3)), 0.5)
    r1.set(composite_domain.index((3, 2)), 0.5)
    r1.set(composite_domain.index((3, 4)), 0.2)
    r1.set(composite_domain.index((4, 3)), 0.2)

    r2 = deepcopy(r1)

    print("Na početku, "
          "relacija {} neizrazita relacija ekvivalencije.".format(conclude(Relations.is_fuzzy_equivalence(r2))))

    for i in range(1, 4):
        print()
        r2 = Relations.binary_relation_composite(r2, r1)

        print("Nakon {} odrađenih kompozicija, relacija glasi:\n{}".format(i, r2))
        print("Takva relacija {} "
              "neizrazita relacija ekvivalencije.".format(conclude(Relations.is_fuzzy_equivalence(r2))))


def do_tasks():
    task_1()
    task_2()
    task_3()


do_tasks()
