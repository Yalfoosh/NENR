import re

import numpy as np


whitespace_regex = re.compile(r"\s+")


def parse_dataset(path: str):
    with open(path) as file:
        lines = [whitespace_regex.split(line) for line in file.readlines()]

        return [(np.array([float(x) for x in line[:2]]), np.array([int(x) for x in line[2:5]])) for line in lines]
