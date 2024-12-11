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

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

__version__ = '0.0.1dev0'

import os
import sys
import pathlib
import warnings

try:
  from lightnumpy.__config__ import show as show_config
except (ImportError, ModuleNotFoundError) as e:
  msg = (
    "Error importing lightnumpy:\n"
    "Cannot import lightnumpy while being in lightnumpy source directory\n"
    "please exit the lightnumpy source tree first and\n"
    "relaunch your Python interpreter."
  )
  # raise ImportError(msg) from e    
  # log.error('BOOM! :: %s', msg)
  # sys.stderr.write('BOOM! :: %s\n' % msg)
  
  show_config = _BUILT_WITH_MESON = None;  del msg;
else:
  _BUILT_WITH_MESON = True

if _BUILT_WITH_MESON:
  from .cy_bindings import cy_numcpp_api; #del cy_bindings;
  from .py_bindings import py_numcpp_api; #del py_bindings;


from .python_api import *; del python_api;

# Remove symbols imported for internal use
del (
  os, sys, pathlib, warnings,
  List, Tuple,
)
