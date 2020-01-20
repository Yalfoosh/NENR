from sys import stdin, stderr

from fuzzy_controller import COA, AccelerationSystem, RudderSystem, product_machine
from util import parse_input_line, stringify_output_line


def main():
    debuzzifier = COA()
    acceleration_system = AccelerationSystem(debuzzifier)
    rudder_system = RudderSystem(debuzzifier)

    values = parse_input_line(stdin.readline())
    # values = (100, 100, 141, 141, 5, 0)

    while values is not None:
        new_acceleration = acceleration_system.conclude(values, product_machine)
        new_angle = rudder_system.conclude(values, product_machine)

        print(stringify_output_line(new_acceleration, new_angle), flush=True)

        input_string = stdin.readline()
        values = parse_input_line(input_string)


main()
