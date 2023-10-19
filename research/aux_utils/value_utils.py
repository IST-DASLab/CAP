from typing import Union


def mean_value(values: Union[int, float, list]):
    if isinstance(values, (int, float)):
        return values
    if isinstance(values, list):
        return sum(values) / len(values)
    else:
        return 0.0
        