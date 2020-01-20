from typing import Tuple

import numpy as np


class DisplayStrategy:
    def encode(self, raw_input, **kwargs):
        raise NotImplemented

    def decode(self, encoded_input, **kwargs):
        raise NotImplemented


class DefaultDisplay(DisplayStrategy):
    def encode(self, raw_input, **kwargs):
        return raw_input

    def decode(self, encoded_input, **kwargs):
        return encoded_input


class BinaryDisplay(DisplayStrategy):
    def __init__(self, value_range: Tuple, bit_count: int):
        self.__value_range = value_range
        self.__bit_count = bit_count

        self.__quantum = (2 ** bit_count - 1) / (value_range[1] - value_range[0])

    def __integer_to_binary(self, value: int):
        binary_representation = np.binary_repr(value)

        count_difference = self.__bit_count - len(binary_representation)

        if count_difference != 0:
            binary_representation = "".join(["0"] * count_difference) + binary_representation

        return np.array([int(element) for element in binary_representation])

    @staticmethod
    def __binary_to_integer(value: np.ndarray):
        to_return = 0
        current_mult = 1

        for element in value[::-1]:
            if element == 1:
                to_return += current_mult

            current_mult *= 2

        return to_return

    @property
    def value_range(self):
        return self.__value_range

    @property
    def bit_count(self):
        return self.__bit_count

    def encode(self, raw_input, **kwargs):
        transformed_input = np.array([int(x + 0.5) for x in ((raw_input - self.value_range[0]) * self.__quantum)])

        return np.array([self.__integer_to_binary(x) for x in transformed_input])

    def decode(self, encoded_input: np.ndarray, **kwargs):
        transformed_input = np.array([self.__binary_to_integer(x) for x in encoded_input])

        return transformed_input / self.__quantum + self.__value_range[0]
