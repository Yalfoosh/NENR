from sys import stdout
from typing import List

from tqdm import tqdm

from civilization.callbacks import Callback
from civilization.initializers import NormalInitializer
from civilization.losses import Loss
from civilization.population import Population
from civilization.signal import Antenna, CallbackID
from civilization.variable import Variable, Constant

from civilization.cross_strategies import CrossStrategy, StochasticCross
from civilization.display_strategies import DisplayStrategy, DefaultDisplay
from civilization.mutation_strategies import MutationStrategy, SwapMutation
from civilization.selection_strategies import SelectionStrategy, TournamentSelection


class Civilization:
    def __init__(self,
                 display_strategy: DisplayStrategy = DefaultDisplay(),
                 selection_strategy: SelectionStrategy = TournamentSelection(),
                 cross_strategy: CrossStrategy = StochasticCross(),
                 mutation_strategy: MutationStrategy = SwapMutation(),
                 **kwargs):
        self._display_strategy = display_strategy
        self._selection_strategy = selection_strategy
        self._cross_strategy = cross_strategy
        self._mutation_strategy = mutation_strategy

        self._initializer_class = kwargs.get("initializer_class", NormalInitializer)

    @property
    def initializer_class(self):
        return self._initializer_class

    @property
    def display_strategy(self):
        return self._display_strategy

    @property
    def selection_strategy(self):
        return self._selection_strategy

    @property
    def cross_strategy(self):
        return self._cross_strategy

    @property
    def mutation_strategy(self):
        return self._mutation_strategy

    def fit(self,
            loss: Loss,
            gene_count: int = 1,
            population_count: Variable = Constant(10),
            epochs: int = 1000,
            callbacks: Callback or List[Callback] = None,
            verbose: int = 1,
            **kwargs):
        kwargs["loss"] = loss
        kwargs["population_count"] = population_count
        kwargs["epochs"] = epochs

        antenna = Antenna([kwargs])

        for callback_id in [CallbackID.TrainingStart, CallbackID.TrainingEnd,
                            CallbackID.IterationStart, CallbackID.IterationEnd]:
            antenna.register_broadcaster(callback_id)

        for strategy in [self.selection_strategy, self.cross_strategy, self.mutation_strategy]:
            strategy.register_to_antenna(antenna)

        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]

        for callback in callbacks:
            callback.register_listeners(antenna)

        kwargs["iteration"] = 0

        initializer = self.initializer_class(count=population_count.get(**kwargs), shape=(gene_count,))
        kwargs["population"] = Population(display_strategy=self.display_strategy,
                                          loss=loss,
                                          storage=initializer.generate(**kwargs))

        if verbose > 1:
            print(f"Starting a civilization with a population of {kwargs['population'].count}.")

        antenna.publish(CallbackID.TrainingStart)

        iteration_range = range(kwargs["epochs"])

        if verbose > 0:
            iteration_range = tqdm(iteration_range, file=stdout)

        try:
            for iteration_index in iteration_range:
                _pop = kwargs["population"]

                diff = _pop.count - kwargs["population_count"].get(**kwargs)

                if diff > 0:
                    _pop.kill_last(diff)

                if verbose > 1:
                    iteration_range.set_description(f"Population: {_pop.count}\t"
                                                    f"Best: {_pop[0]}")

                kwargs["iteration"] = iteration_index

                antenna.publish(CallbackID.IterationStart)

                kwargs["population"], selection = self.selection_strategy.select(pop=kwargs["population"], **kwargs)
                children = self.cross_strategy.cross(parents=selection, **kwargs)
                children = [self.mutation_strategy.mutate(child, **kwargs) for child in children]

                for child in children:
                    kwargs["population"].add(child)

                antenna.publish(CallbackID.IterationEnd)
        except RuntimeError:
            if verbose > 1:
                print(f"\n\n[Civilization]\tTraining interrupted!")

        antenna.publish(CallbackID.TrainingEnd)

        return kwargs["population"][0]
