"""
LightNumPy - A lightweight, modular numerical computation library.

This package provides basic numerical operations and optimizations, including support for GPU and TPU operations.

Modules:
--------
- python_api: The primary API module, including core operations and GPU/TPU enhancements.
- core: Basic numerical operations such as addition, subtraction, etc.
- gpu_operations: GPU-accelerated operations for numerical calculations.
- tpu_operations: TPU-accelerated operations for numerical calculations.

Example Usage:
--------------
>>> from lightnumpy import python_api
>>> result = python_api.core.add([1, 2], [3, 4])
>>> print(result)
[4, 6]

Notes:
------
This package aims to provide a lightweight alternative to NumPy with support for GPUs and TPUs, making it a flexible solution for numerical computations.

See Also:
----------
- NumPy: https://numpy.org/
"""

__version__ = '0.1.0'

# from .python_api import (
# 	core,
# 	gpu_operations,
# 	tpu_operations,
# )
