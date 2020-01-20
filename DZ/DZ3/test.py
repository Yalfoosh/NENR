from fuzzy_controller import COA, AccelerationSystem, RudderSystem
from util import parse_input_line


def test_1():
    defuzzifier = COA()
    fs = AccelerationSystem(defuzzifier)

    inputs = ""
    while inputs != "A" and inputs != "K":
        inputs = input("A ili K: ").strip()

        if inputs == "K":
            fs = RudderSystem(defuzzifier)
    inputs = ""
    max_rule_count = fs.rule_count() - 1

    while not isinstance(inputs, int):
        inputs = input("Unesite broj pravila ({} do {}): ".format(0, max_rule_count)).strip()

        try:
            inputs = int(inputs)

            if inputs < 0 or inputs > max_rule_count:
                raise ValueError
        except:
            print("Krivi unos.")

    rule_to_check = fs.get_rule(inputs)
    inputs = ""

    while not isinstance(inputs, list):
        try:
            inputs = parse_input_line(input("Unesite L, D, LK, DK, V, S: ").strip())

            if inputs is None:
                raise ValueError
        except:
            print("Krivi unos.")

    a = rule_to_check.conclude(inputs)
    a_defuzz = defuzzifier.defuzzify(a)

    print("Uz unos {} zaključujemo:\n{}".format(inputs, a))
    print("Kad se neizrazitost dekodira, dobijemo: {}".format(a_defuzz))


def test_2():
    defuzzifier = COA()
    fs = AccelerationSystem(defuzzifier)
    rs = RudderSystem(defuzzifier)

    inputs = ""
    while not isinstance(inputs, list):
        try:
            inputs = parse_input_line(input("Unesite L, D, LK, DK, V, S: ").strip())

            if inputs is None:
                raise ValueError
        except:
            print("Krivi unos.")

    a = fs.infer(inputs)
    print("Unija zaključaka akceleracije:\n{}\nDekodirani zaključak: {}".format(a, defuzzifier.defuzzify(a)))

    a = rs.infer(inputs)
    print("Unija zaključaka kormila:\n{}\nDekodirani zaključak: {}".format(a, defuzzifier.defuzzify(a)))


test_2()
