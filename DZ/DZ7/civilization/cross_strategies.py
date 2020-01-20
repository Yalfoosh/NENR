from typing import List

import numpy as np

from civilization.losses import Loss
from civilization.signal import Antenna, CallbackID
from civilization.variable import Variable, Constant


class CrossStrategy:
    def __init__(self, keep_parents: bool = False):
        self.__keep_parents = keep_parents

        self.__antenna = None

    @property
    def keep_parents(self):
        return self.__keep_parents

    def cross(self, parents: List[np.ndarray], **kwargs):
        if self.__antenna is not None:
            self.__antenna.publish(CallbackID.BeforeCross)

        survivors = parents if self.keep_parents else []
        survivors.extend(self._cross(parents, **kwargs))

        if self.__antenna is not None:
            self.__antenna.publish(CallbackID.AfterCross)

        return survivors

    def _cross(self, parents: List[np.ndarray], **kwargs):
        raise NotImplementedError

    @staticmethod
    def _validate_parents(parents: List[np.ndarray]):
        for i, parent in enumerate(parents):
            if not isinstance(parent, np.ndarray):
                raise RuntimeError(f"Parent {i} should be a numpy.ndarray (it is currently {type(parent)})!")

        parent_shape = parents[0].shape

        if not np.all([parent.shape == parent_shape for parent in parents]):
            raise RuntimeError(f"Not all parents have identical shapes (need to have shape {parent_shape}) "
                               f"-> can't cross!")

    @staticmethod
    def _fix_parents(parents: List[np.ndarray], maximum_parent_count: int = 2):
        if not len(parents) % maximum_parent_count == 0:
            raise RuntimeError(f"Stochastic cross is not meant for a parent count not divisible by "
                               f"len(parents) ({len(parents)} !% {maximum_parent_count})!")

        parents = [parents[i: i + maximum_parent_count] for i in range(len(parents) // maximum_parent_count)]

        return parents

    def register_to_antenna(self, antenna: Antenna):
        self.__antenna = antenna

        for callback_id in [CallbackID.BeforeCross, CallbackID.AfterCross]:
            antenna.register_broadcaster(callback_id)


class StochasticCross(CrossStrategy):
    def __init__(self, probability: Variable = Constant(0.5), keep_parents: bool = False):
        super().__init__(keep_parents=keep_parents)

        self.__probability = probability

    @property
    def probability(self):
        return self.__probability

    def _cross(self, parents: List[np.ndarray], **kwargs):
        CrossStrategy._validate_parents(parents)

        children = list()

        for mother, father in CrossStrategy._fix_parents(parents, 2):
            child = np.copy(father)

            for i, gene in enumerate(mother):
                if np.random.uniform() < self.probability.get(**kwargs):
                    child[i] = gene

            children.append(child)

        return children


class AveragingCross(CrossStrategy):
    def __init__(self, parent_count: Variable = Constant(2), keep_parents: bool = False):
        super().__init__(keep_parents=keep_parents)

        self.__parent_count = parent_count

    @property
    def parent_count(self):
        return self.__parent_count

    def _cross(self, parents, **kwargs):
        CrossStrategy._validate_parents(parents)

        children = list()

        for parents in CrossStrategy._fix_parents(parents, self.parent_count.get(**kwargs)):
            children.append(np.mean(np.array(parents), axis=0))

        return children


class BreakingPointCross(CrossStrategy):
    def __init__(self, loss: Loss, keep_parents: bool = False):
        super().__init__(keep_parents=keep_parents)

        self.__loss = loss

    @property
    def loss(self):
        return self.__loss

    def _cross(self, parents, **kwargs):
        CrossStrategy._validate_parents(parents)

        children = list()

        for mother, father in CrossStrategy._fix_parents(parents, 2):
            breaking_point_index = np.random.choice(len(mother) - 1)

            c_children = [np.concatenate([mother[: breaking_point_index]], father[breaking_point_index:]),
                          np.concatenate([father[: breaking_point_index]], mother[breaking_point_index:])]

            if self.loss is None or not isinstance(self.loss, Loss):
                children.extend(c_children)
            else:
                children.append(c_children[int(np.argmin([self.loss.call(child, **kwargs) for child in c_children]))])

        return children
