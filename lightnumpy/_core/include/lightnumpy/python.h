#pragma once
// Notes:
// - To use the Python C API, you must initialize the Python interpreter using `Py_Initialize()`.
// - After you're done with Python, call `Py_Finalize()` to clean up and shut down the interpreter.
// - Common Python C API functions include `PyRun_SimpleString()` (to execute Python code) and `PyImport_ImportModule()` (to import Python modules).
// - Python objects are handled using `PyObject*` types, and you can convert between C data types and Python objects using functions like `Py_BuildValue()` and `PyArg_ParseTuple()`.
// - Always manage reference counts properly using `Py_INCREF()` and `Py_DECREF()` to avoid memory leaks.

// Python C API header
// This allows C/C++ code to interact with Python, enabling Python script execution and data manipulation within a C/C++ program.
#include <Python.h>

// #error "Python headers are missing. Ensure you have Python development headers installed."