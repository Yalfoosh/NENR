import numpy as np
import matplotlib.pyplot as plt


def the_function(inputs):
    t = \
        [
            np.square(inputs[0] - 1),
            np.square(inputs[1] + 2),
            -5 * inputs[0] * inputs[1],
            3,
            np.square(np.cos(inputs[0] / 5))
        ]

    return (t[0] + t[1] + t[2] + t[3]) * t[4]


def split_data_into_batches(dataset, batch_size):
    no_batches = len(dataset) // batch_size
    part_size = len(dataset) % batch_size
    full_size = batch_size * no_batches

    dataset_full = dataset[: full_size].reshape((no_batches, batch_size, *dataset[0].shape))
    dataset_part = np.array([])

    if part_size != 0:
        dataset_part = dataset[full_size:].reshape((1, part_size, *dataset[0].shape))

    dataset = list(dataset_full)

    if part_size != 0:
        dataset.extend(list(dataset_part))

    return dataset


def generate_anfis_dataset(x_range=(-4, 4 + 1), y_range=(-4, 4 + 1)):
    xs, ys = list(), list()

    for x in range(*x_range):
        for y in range(*y_range):
            inputs = np.array([x, y])

            xs.append(inputs)
            ys.append(the_function(inputs))

    return xs, ys
