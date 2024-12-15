#pragma once
// Notes:
// - Pybind11 simplifies the process of creating Python extensions by eliminating the need for boilerplate code. With Pybind11, you can directly expose C++ classes and functions to Python.
// - Pybind11 supports both Python 2 and 3, and it works with modern C++ features, such as lambdas and smart pointers.
// - To expose a C++ class or function to Python, you only need to write a minimal wrapper, usually in a `.cpp` file, using `PYBIND11_MODULE()` to define the Python module.
// - Example: `PYBIND11_MODULE(my_module, m) { m.def("my_function", &my_function); }` binds the C++ function `my_function` to the Python module `my_module`.
// - Pybind11 handles the conversion between C++ types and Python types (e.g., `std::string` becomes `str`, `int` becomes `int`, etc.) automatically.

// Optional Pybind11 header
// Pybind11 is a lightweight header-only library for creating Python bindings to C++ code.
// It allows C++ classes and functions to be directly exposed to Python.
#ifdef PYBIND11_H_INCLUDED  // if defined(PYBIND11_H_INCLUDED)
    #include <pybind11/pybind11.h>
#else
    // ifndef PYBIND11_H_INCLUDED  // if !defined(PYBIND11_H_INCLUDED)
    // Pybind11 is optional, so no error is raised.
    // If you need Pybind11, install it and ensure it's available.
#endif
