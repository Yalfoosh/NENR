from ast import literal_eval
import csv
import numpy as np
import re

decimal_regex = re.compile(r"\d+\.\d+")


def load_dataset(path: str):
    with open(path) as file:
        reader = csv.reader(file, delimiter="\t")

        dataset = list()

        for row in reader:
            if row is None or len(row) == 0:
                continue

            dataset.append((np.array(literal_eval(row[0])), literal_eval(row[1])))

    return dataset


def clip_dataset(dataset):
    to_return = list()

    for x, y in dataset:
        gesture = np.array(x, dtype="float")
        gesture -= np.mean(gesture, axis=0)
        gesture /= np.max(np.abs(gesture))

        to_return.append((gesture, y))

    return to_return


def lerp(point_1: np.ndarray, point_2: np.ndarray, amount: float = 0.5):
    direction = point_2 - point_1
    vector = amount * direction
    
    return point_1 + vector


def interpolate_dataset(dataset, number_of_final_points: int = 20, epsilon: float = 1e-6):
    to_return = list()

    for x, y in dataset:
        if number_of_final_points == 1:
            to_return.append((x[0], y))
        else:
            total_distance = 0
            last_coordinate = x[0]
            distances = [0.]

            for point in (x[1:]):
                total_distance += np.linalg.norm(point - last_coordinate)
                last_coordinate = point
                distances.append(total_distance)

            new_x = list()

            last_visited_index = 0
            current_distance = distances[last_visited_index]
            for i in range(number_of_final_points):
                goal_distance = (float(i) / (number_of_final_points - 1)) * total_distance
                difference = goal_distance - current_distance

                while difference > epsilon:
                    last_visited_index += 1
                    current_distance = distances[last_visited_index]
                    difference = goal_distance - current_distance

                if difference > 0.:
                    new_x.append(x[last_visited_index])
                else:
                    neighbouring_difference = np.linalg.norm(x[last_visited_index] - x[last_visited_index - 1])
                    ratio = 1. + (difference / neighbouring_difference)

                    new_x.append(lerp(x[last_visited_index - 1], x[last_visited_index], ratio))

            to_return.append((np.array(new_x), y))

    return to_return


def flatten_dataset(dataset):
    return [(np.reshape(x[0], (-1, )), x[1]) for x in dataset]


def discrete_to_onehot_dataset(dataset, ndims: int = 5):
    to_return = list()

    for x, y in dataset:
        onehot = np.zeros(ndims)
        onehot[y] = 1.

        to_return.append((x, onehot))

    return to_return


def normalize_dataset(dataset, number_of_final_points: int = 20, epsilon: float = 1e-6, class_count: int = 5):
    return discrete_to_onehot_dataset(flatten_dataset(interpolate_dataset(clip_dataset(dataset),
                                                                          number_of_final_points,
                                                                          epsilon)), class_count)


def normalize_gesture(gesture, number_of_final_points: int = 20, epsilon: float = 1e-6, class_count: int = 5):
    gesture_with_dummy = [gesture, 0]
    dummy_dataset = [gesture_with_dummy]

    normalized_dummy = normalize_dataset(dummy_dataset, number_of_final_points, epsilon, class_count)

    return normalized_dummy[0][0]
