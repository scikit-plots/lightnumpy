"""
Core operations for numerical computation.

This module contains the basic operations that are fundamental to numerical computations.

Functions:
----------
- add: Adds two lists element-wise.
- subtract: Subtracts two lists element-wise.

Example Usage:
--------------
>>> from lightnumpy.python_api import core
>>> result = core.add([1, 2], [3, 4])
>>> print(result)
[4, 6]
"""
# Template: Define core functionality for array manipulation.

from typing import List

def add(a: List[float], b: List[float]) -> List[float]:
    """
    Adds two lists element-wise.

    Args:
        a: First list of numbers.
        b: Second list of numbers.

    Returns:
        A list containing the element-wise sum of the two lists.

    Example:
    >>> add([1, 2], [3, 4])
    [4, 6]
    """
    return [x + y for x, y in zip(a, b)]

def subtract(a: List[float], b: List[float]) -> List[float]:
    """
    Subtracts two lists element-wise.

    Args:
        a: First list of numbers.
        b: Second list of numbers.

    Returns:
        A list containing the element-wise difference of the two lists.

    Example:
    >>> subtract([3, 4], [1, 2])
    [2, 2]
    """
    return [x - y for x, y in zip(a, b)]
