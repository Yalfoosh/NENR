from typing import Iterable, Tuple

import numpy as np
from sortedcontainers import SortedList

from civilization.losses import Loss

from civilization.display_strategies import DisplayStrategy


class Population:
    def __init__(self, display_strategy: DisplayStrategy, loss: Loss, storage: Iterable[np.ndarray] = None):
        self.__loss = loss

        storage = [display_strategy.encode(element) for element in storage]
        self.__storage = SortedList([(self.loss.call(element), element) for element in storage]
                                    if storage is not None else [], key=lambda x: x[0])

    @property
    def loss(self):
        return self.__loss

    @property
    def count(self):
        return len(self.__storage)

    def add(self, specimen: np.ndarray):
        self.add_raw(entry=(self.loss.call(specimen), specimen))

    def add_raw(self, entry: Tuple[float, np.ndarray]):
        self.__storage.add(entry)

    def kill(self, index: int = -1):
        del self.__storage[index]

    def kill_many(self, indices):
        count = 0

        for index in sorted(set(indices)):
            del self.__storage[index - count]
            count += 1

    def kill_last(self, count: int = 1):
        if count < 0:
            raise RuntimeError(f"Can't kill the last {count} specimens, it's a negative number!")

        count = min(len(self.__storage), count)

        for _ in range(count):
            self.kill()

    def pop(self, index: int = -1):
        self.__storage.pop(index)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.__storage[key]
        elif isinstance(key, slice):
            return self.__storage[key.start: key.stop: key.step]
        else:
            to_return = list()

            for entry in key:
                to_return.append(self.__storage[entry])

            return to_return
