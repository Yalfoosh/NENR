from collections import defaultdict
from genetics import Civilization
from util import parse,\
                 constant, linear_iteration_decay, explicit,\
                 generative_select, three_tournament_select,\
                 mean_cross,\
                 gaussian_mutate

# Rješenje 1. [0.3712056, -0.12289254, 3.5152062, 1.31750053, -1.32588974] (MSE = 6e-7, 1000 iteracija)
# Rješenje 2. [0.37028343, -0.11768803, 3.46963318, 1.31324859, -1.32198286] (MSE = 1e-2, 1000 iteracija)

first_data_path = "dataset_1.txt"
second_data_path = "dataset_2.txt"

training_set = parse(first_data_path)
number_of_genes = 5
gene_interval = (-4, 4)

population_function = linear_iteration_decay(20, 1/10, lower_bound=7)

select_function = three_tournament_select
mortality = linear_iteration_decay(3, -1/30, lower_bound=1)

crossover_function = mean_cross

mutate_function = gaussian_mutate
mutation_probability = linear_iteration_decay(0.01, -1/1000, upper_bound=0.1)

intensity_map = dict()

for i in range(0, 100):
    intensity_map[i] = 1

for i in range(100, 400):
    intensity_map[i] = 1.5

mutation_intensity = explicit(0.1, intensity_map)


iteration_count = 1000

civilization = Civilization(training_set=training_set,
                            gene_count=number_of_genes,
                            gene_interval=gene_interval,
                            population_function=population_function,

                            select_function=select_function,
                            cross_function=crossover_function,
                            mutate_function=mutate_function)


best_result = civilization.evolve(iterations=iteration_count, verbose=2,
                                  mortality=mortality,
                                  gauss_probability=mutation_probability,
                                  gauss_intensity=mutation_intensity)

from util import the_function

print(f"Loss(best) = {civilization.loss_function(best_result)}\t\t{best_result}")

for entry in training_set:
    print(f"{the_function(*entry[0], best_result):.3f} (instead of {entry[1]:.3f})")
