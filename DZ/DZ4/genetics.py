from typing import Any, Callable, List, Tuple
import numpy as np

from util import constant, three_tournament_select, stochastic_cross, stochastic_swap_mutate, the_function, mse,\
                 chromosome_loss


class Civilization:
    def __init__(self, training_set: List[Tuple[Any, Any]],
                 gene_count: int = 5, gene_interval=(0., 1.),
                 population_function: Callable = constant(50),
                 select_function: Callable = three_tournament_select,
                 cross_function: Callable = stochastic_cross,
                 mutate_function: Callable = stochastic_swap_mutate,
                 prediction_function: Callable = the_function,
                 loss_function: Callable = mse,
                 loss_to_wellness: Callable = lambda x: np.reciprocal(x), **kwargs):
        self.state = dict()
        self.train_x, self.train_y = zip(*training_set)

        self.population_function = population_function
        self.__population = list()

        a, b = gene_interval

        for _ in range(int(self.population_function(iteration=0))):
            self.__population.append(np.add(a, np.multiply(np.subtract(b, a), np.random.rand(gene_count))))

        self.select = select_function
        self.cross = cross_function
        self.mutate = mutate_function

        self.prediction_function = prediction_function
        self.loss_function = lambda c: chromosome_loss(x=self.train_x,
                                                       y_pred=self.train_y,
                                                       chromosome=c,
                                                       prediction_function=self.prediction_function,
                                                       loss_function=loss_function)
        self.wellness_function = lambda x: loss_to_wellness(self.loss_function(x))

        self.state = kwargs
        self.state["gene_count"] = gene_count
        self.state["gene_interval"] = gene_interval
        self.state["population"] = self.population
        self.state["max_population"] = int(self.population_function(iteration=0))
        self.state["iteration"] = 0
        self.state["prediction_function"] = self.prediction_function
        self.state["loss_function"] = self.loss_function
        self.state["wellness_function"] = self.wellness_function

    @property
    def population(self):
        return self.__population

    def wellness(self, chromosome: np.ndarray):
        return self.wellness_function(chromosome)

    def evolve(self, iterations: int, verbose: int = 1, **kwargs):
        kwargs.update(self.state)
        self.__population = sorted(self.population, key=lambda x: self.wellness(x), reverse=True)

        best_one = self.population[0]

        if verbose > 0:
            print(f"Najbolji primjerak ima dobrotu {self.wellness(best_one)}.")

        for i in range(iterations):
            kwargs["iteration"] = i

            if len(self.population) > kwargs["max_population"]:
                self.__population = self.population[:kwargs["max_population"]]

            kwargs["population"] = self.population
            kwargs["max_population"] = int(self.population_function(**kwargs))

            new_population, parent_list = self.select(**kwargs)

            for parents in parent_list:
                new_population.append(self.mutate(self.cross(parents, **kwargs), **kwargs))

            self.__population = sorted(new_population, key=lambda x: self.wellness_function(x), reverse=True)

            if self.wellness(self.__population[0]) > self.wellness(best_one):
                best_one = self.__population[0]

                if verbose > 0:
                    print(f"[Iteracija {i}] - Novi najbolji primjerak (dobrota {self.wellness(best_one):.3f}): "
                          f"{best_one}.")

            if i % 10 == 0:
                print(f"\n-------------------------------------- Iteracija {i} --------------------------------------")

                if verbose > 0:
                    print(f"\nNajbolji primjerak of njih {len(self.population)} u {i}. iteraciji je {best_one}: "
                          f"dobrota = {self.wellness(best_one)}.")

                if verbose > 1:
                    print("\nNajboljih 5 pripadnika populacije:")

                    for citizen in self.population[:5]:
                        print(f"\t{self.wellness(citizen):.6f}\t\t{citizen}")

                print(f"\n-------------------------------------------------------------------------------------------")

        return best_one
