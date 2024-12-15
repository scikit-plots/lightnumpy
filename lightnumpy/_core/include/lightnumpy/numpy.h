#pragma once
// Notes:
// - NumPy arrays are represented as `PyArrayObject*` in the C API.
// - To work with NumPy arrays, you need to initialize the NumPy API by calling `import_array()` (usually done in the `main()` function).
// - NumPy provides a range of functions for array manipulation, such as `PyArray_Alloc()` (for creating arrays) and `PyArray_GetPtr()` (to access elements).
// - The NumPy C API allows for high-performance numerical operations, making it ideal for scientific computing.
// - Make sure to include `numpy/arrayobject.h` in your project and link the appropriate NumPy C library.

// NumPy C API headers
// https://pythonextensionpatterns.readthedocs.io/en/latest/cpp_and_numpy.html
// This enables access to NumPy arrays and operations in C/C++ code. NumPy is widely used for numerical computations and large datasets in Python.
#include <numpy/arrayobject.h>

// #error "NumPy headers are missing. Ensure NumPy development headers are installed."