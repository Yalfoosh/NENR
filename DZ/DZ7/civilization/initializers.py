from typing import Tuple

import numpy as np


class Initializer:
    def __init__(self, count: int, shape: Tuple):
        self.__count = count
        self.__shape = shape

    @property
    def count(self):
        return self.__count

    @property
    def shape(self):
        return self.__shape

    def generate(self, **kwargs):
        raise NotImplemented


class NormalInitializer(Initializer):
    def generate(self, **kwargs):
        return [x for x in np.random.normal(loc=kwargs.get("loc", 0.), scale=kwargs.get("scale", 1.),
                                            size=(self.count, *self.shape))]


class UniformInitializer(Initializer):
    def generate(self, **kwargs):
        return [x for x in np.random.uniform(low=kwargs.get("low", -1.),
                                             high=kwargs.get("high", 1.),
                                             size=(self.count, *self.shape))]
