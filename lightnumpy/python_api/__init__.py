"""
LightNumPy Python API - A Python API for basic numerical and optimized operations.

This module provides basic operations (such as addition and subtraction) and optimized implementations for GPU and TPU.

Modules:
--------
- core: Core operations (e.g., addition, subtraction).
- gpu_operations: Operations optimized for GPU execution.
- tpu_operations: Operations optimized for TPU execution.

Example Usage:
--------------
>>> from lightnumpy.python_api import core
>>> result = core.add([1, 2], [3, 4])
>>> print(result)
[4, 6]

Notes:
------
The core module includes standard numerical operations, while the GPU and TPU operations offer optimizations for high-performance computing hardware.

See Also:
----------
- NumPy: https://numpy.org/
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Template: The main entry point for python_api.

from ._utils_impl import *

from . import (
  array,
  core,
)