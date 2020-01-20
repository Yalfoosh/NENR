import re
from typing import Tuple


def parse_input_line(string: str) -> Tuple[int, int, int, int, int, int] or None:
    if re.match(r"\s*[Kk]\s*[Rr]\s*[Aa]\s*[Jj]\s*", string):
        return None

    elements = list(map(lambda x: int(x.strip()),
                        re.split(r"\s+", string.strip(), 5)))

    return elements


def stringify_output_line(acceleration: int, angle: int) -> str:
    return "{} {}".format(acceleration, angle)
