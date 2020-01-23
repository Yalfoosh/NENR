from typing import List, Tuple

import numpy as np

import minikeras.layers
import minikeras.losses
import minikeras.model

import civilization.callbacks
import civilization.civilization
import civilization.cross_strategies
import civilization.display_strategies
import civilization.initializers
import civilization.losses
import civilization.mutation_strategies
import civilization.selection_strategies
import civilization.variable

from util import parse_dataset


default_composite_mutations =\
    [
        civilization.mutation_strategies.GaussianMutation(intensity=civilization.variable.Constant(5.)),
        civilization.mutation_strategies.GaussianMutation(intensity=civilization.variable.Constant(50.)),
        civilization.mutation_strategies.GaussianReplacementMutation()
    ]

default_composite_probabilities = np.array([10., 2., 1.])

tr_x, tr_y = zip(*parse_dataset("data/dataset.tsv"))


class GeneticNN:
    def __init__(self, sizes: List[int] or Tuple,
                 selection_strategy: civilization.selection_strategies.SelectionStrategy = None,
                 cross_strategy: civilization.cross_strategies.CrossStrategy = None,
                 mutation_strategy: civilization.mutation_strategies.MutationStrategy = None,
                 initializer_class=None):
        if selection_strategy is None:
            selection_strategy = civilization.selection_strategies.TournamentSelection()

        if cross_strategy is None:
            cross_strategy = civilization.cross_strategies.AveragingCross()

        if mutation_strategy is None:
            mutation_strategy = civilization.mutation_strategies.CompositeMutation(default_composite_mutations,
                                                                                   default_composite_probabilities)

        if initializer_class is None:
            initializer_class = civilization.initializers.UniformInitializer

        display_strategy = civilization.display_strategies.DefaultDisplay()

        layers = [minikeras.layers.Input(sizes[0]),
                  minikeras.layers.Distance(sizes[1])]

        for size in sizes[2:]:
            layers.append(minikeras.layers.Dense(size, activation="sigmoid"))

        self._model = minikeras.model.Model(layers)
        self._model.compile(loss=minikeras.losses.AbsoluteError())

        self._civilization = civilization.civilization.Civilization(display_strategy=display_strategy,
                                                                    selection_strategy=selection_strategy,
                                                                    cross_strategy=cross_strategy,
                                                                    mutation_strategy=mutation_strategy,
                                                                    initializer_class=initializer_class)

        self._genetic_loss = civilization.losses.DatasetNNMeanSquareError(self._model,
                                                                          (tr_x, tr_y),
                                                                          display_strategy)

    @property
    def model(self):
        return self._model

    def predict(self, inputs: np.ndarray):
        return np.array([0 if x < 0.5 else 1 for x in self._model.predict(inputs)])

    def fit(self,
            population_count: civilization.variable.Variable = civilization.variable.Constant(100),
            epochs: int = 1000,
            callbacks: civilization.callbacks.Callback or List[civilization.callbacks.Callback] = None,
            verbose: int = 1,
            **kwargs):
        return self._civilization.fit(loss=self._genetic_loss,
                                      gene_count=self._model.extract_parameters().size,
                                      population_count=population_count,
                                      epochs=epochs,
                                      callbacks=callbacks,
                                      verbose=verbose,
                                      low=0.5,
                                      high=1.5,
                                      **kwargs)
