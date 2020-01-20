import numpy as np

from civilization.signal import Antenna, CallbackID
from civilization.variable import Variable, Constant


class MutationStrategy:
    def __init__(self):
        self.__antenna = None

    def mutate(self, specimen: np.ndarray, **kwargs):
        if self.__antenna is not None:
            self.__antenna.publish(CallbackID.BeforeMutation)

        result = self._mutate(specimen, **kwargs)

        if self.__antenna is not None:
            self.__antenna.publish(CallbackID.AfterMutation)

        return result

    def _mutate(self, specimen: np.ndarray, **kwargs):
        return np.copy(specimen)

    def register_to_antenna(self, antenna: Antenna):
        self.__antenna = antenna

        for callback_id in [CallbackID.BeforeMutation, CallbackID.AfterMutation]:
            antenna.register_broadcaster(callback_id)


class SwapMutation(MutationStrategy):
    def __init__(self, probability: Variable = Constant(0.01)):
        super().__init__()

        self.__probability = probability

    @property
    def probability(self):
        return self.__probability

    def _mutate(self, specimen: np.ndarray, **kwargs):
        new_specimen = specimen.copy()
        max_count = len(new_specimen)

        count = 0
        while count < max_count and np.random.uniform() < self.probability.get(**kwargs):
            indices = np.random.choice(np.arange(len(new_specimen)), size=2)

            new_specimen[indices[0]], new_specimen[indices[1]] = new_specimen[indices[1]], new_specimen[indices[0]]
            count += 1

        return new_specimen


class GaussianMutation(MutationStrategy):
    def __init__(self, probability: Variable = Constant(0.01), intensity: Variable = Constant(1.)):
        super().__init__()

        self.__probability = probability
        self.__intensity = intensity

    @property
    def probability(self):
        return self.__probability

    @property
    def intensity(self):
        return self.__intensity

    def _mutate(self, specimen: np.ndarray, **kwargs):
        new_specimen = np.copy(specimen)

        for i in range(len(new_specimen)):
            if np.random.uniform() < self.probability.get(**kwargs):
                new_specimen[i] = np.random.normal(loc=new_specimen[i], scale=self.intensity.get(**kwargs))

        return new_specimen


class GaussianReplacementMutation(MutationStrategy):
    def __init__(self, probability: Variable = Constant(0.01), intensity: Variable = Constant(1.)):
        super().__init__()

        self.__probability = probability
        self.__intensity = intensity

    @property
    def probability(self):
        return self.__probability

    @property
    def intensity(self):
        return self.__intensity

    def _mutate(self, specimen: np.ndarray, **kwargs):
        new_specimen = np.copy(specimen)

        for i in range(len(new_specimen)):
            if np.random.uniform() < self.probability.get(**kwargs):
                new_specimen[i] = np.random.normal(loc=0, scale=self.intensity.get(**kwargs))

        return new_specimen


class CompositeMutation(MutationStrategy):
    def __init__(self, mutation_strategies, probabilities = None):
        super().__init__()

        if probabilities is None:
            probabilities = np.array([1.] * len(mutation_strategies))

        self.__mutation_strategies = mutation_strategies
        self.__probabilities = np.array(probabilities)
        self.__probabilities = self.__probabilities / np.sum(probabilities)

        self.__choice_array = np.arange(len(self.mutation_strategies))

    @property
    def mutation_strategies(self):
        return self.__mutation_strategies

    @property
    def probabilities(self):
        return self.__probabilities

    def _mutate(self, specimen: np.ndarray, **kwargs):
        return self.mutation_strategies[np.random.choice(self.__choice_array, p=self.probabilities)]\
                   .mutate(specimen, **kwargs)
