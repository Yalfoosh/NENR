import numpy as np

from civilization.population import Population
from civilization.signal import Antenna, CallbackID
from civilization.variable import Variable, Constant


class SelectionStrategy:
    def __init__(self):
        self.__antenna = None

    def select(self, pop, **kwargs):
        if self.__antenna is not None:
            self.__antenna.publish(CallbackID.BeforeSelection)

        result = self._select(pop, **kwargs)

        if self.__antenna is not None:
            self.__antenna.publish(CallbackID.AfterSelection)

        return result

    def _select(self, pop, **kwargs):
        raise NotImplementedError

    def register_to_antenna(self, antenna: Antenna):
        self.__antenna = antenna

        for callback_id in [CallbackID.BeforeSelection, CallbackID.AfterSelection]:
            antenna.register_broadcaster(callback_id)


class RouletteSelection(SelectionStrategy):
    def __init__(self, fertility: Variable = Constant(1), mortality: Variable = Constant(1),
                 parent_batch_size: Variable = Constant(2)):
        super().__init__()

        self.__fertility = fertility
        self.__mortality = mortality
        self.__parent_batch_size = parent_batch_size

    @property
    def fertility(self):
        return self.__fertility

    @property
    def mortality(self):
        return self.__mortality

    @property
    def parent_batch_size(self):
        return self.__parent_batch_size

    def _select(self, pop: Population, **kwargs):
        fertility = self.fertility.get(**kwargs)
        mortality = self.mortality.get(**kwargs)
        parent_batch_size = self.parent_batch_size.get(**kwargs)

        fitnesses = [pop.loss.fitness(x[1], loss_value=x[0]) for x in pop]
        logits = pop.loss.logits(fitnesses, **kwargs)

        chosen_indices = np.random.choice(np.arange(pop.count), size=(parent_batch_size * fertility), p=logits)
        selected = pop.pop(chosen_indices)
        pop.kill_many(chosen_indices)

        if mortality > 0:
            pop.kill_many(np.arange(pop.count - min(pop.count, mortality), pop.count))

        return pop, selected


class TournamentSelection(SelectionStrategy):
    def __init__(self, fertility: Variable = Constant(1), mortality: Variable = Constant(1),
                 k: Variable = Constant(3), p: Variable = Constant(0.5),
                 parent_batch_size: Variable = Constant(2), choice_per_tournament: Variable = Constant(1)):
        super().__init__()

        self.__fertility = fertility
        self.__mortality = mortality
        self.__k = k
        self.__p = p
        self.__parent_batch_size = parent_batch_size
        self.__choice_per_tournament = choice_per_tournament

    @property
    def fertility(self):
        return self.__fertility

    @property
    def mortality(self):
        return self.__mortality

    @property
    def k(self):
        return self.__k

    @property
    def p(self):
        return self.__p

    @property
    def parent_batch_size(self):
        return self.__parent_batch_size

    @property
    def choice_per_tournament(self):
        return self.__choice_per_tournament

    def _select(self, pop, **kwargs):
        """ Do fertility * parent_batch_size times: Choose k participants with equal probabilities, one wins with
            probabilities in order from the best unit: [p, p(1-p), p(1-p)^2, ...]."""
        fertility = self.fertility.get(**kwargs)
        mortality = self.mortality.get(**kwargs)
        k = self.k.get(**kwargs)
        p = self.p.get(**kwargs)
        parent_batch_size = self.parent_batch_size.get(**kwargs)
        choice_per_tournament = self.choice_per_tournament.get(**kwargs)

        if parent_batch_size % choice_per_tournament != 0:
            raise RuntimeError(f"Number of choices per tournament ({choice_per_tournament}) should be a common divisor "
                               f"of parent_batch_size ({parent_batch_size}) "
                               f"({parent_batch_size} % {choice_per_tournament} != 0)!")

        chosen = list()

        for _ in range(fertility * parent_batch_size // choice_per_tournament):
            chosen_indices = sorted(np.random.choice(np.arange(pop.count), k))

            selected = [x[1] for x in pop[chosen_indices]]
            selection_probability = [p]

            while len(selection_probability) < len(selected) - 1:
                selection_probability.append(selection_probability[-1] * (1. - selection_probability[0]))

            selection_probability.append(1. - np.sum(selection_probability))

            winner_index = np.random.choice(np.arange(len(selected)), choice_per_tournament, p=selection_probability)
            chosen.extend(np.array(selected)[winner_index])

        if mortality > 0:
            pop.kill_many(np.arange(pop.count - min(pop.count, mortality), pop.count))

        return pop, chosen

