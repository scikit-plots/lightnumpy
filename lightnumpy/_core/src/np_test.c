// src/np_test.cpp
// C++ Source File for initializing Python and NumPy and running a simple test

#pragma once  // Ensures the file is included only once by the compiler

#ifndef NP_TEST_CPP  // Include guard to prevent multiple inclusions
#define NP_TEST_CPP

// Standard C++ library headers
#include <iostream>  // For outputting to the console

// Include necessary NumPy and Python headers
#include "numpy.h"  // NumPy bindings
#include "python.h"  // Python bindings

// Python and NumPy initialization function
int np_test() {
    try {
        // Initialize the Python interpreter
        Py_Initialize();

        // Check if Python is initialized properly
        if (!Py_IsInitialized()) {
            throw std::runtime_error("Python initialization failed.");
        }

        // Initialize the NumPy library (requires import_array from NumPy C-API)
        import_array();

        // Check if NumPy has been initialized properly
        // This is a basic check, more advanced checks can be added if necessary
        if (!PyArray_API) {
            throw std::runtime_error("NumPy initialization failed.");
        }

        // Output success message
        std::cout << "NumPy Initialized Successfully." << std::endl;

        // Finalize the Python interpreter
        Py_Finalize();
    } catch (const std::exception& e) {
        // Print any errors that occur during initialization
        std::cerr << "Error during NumPy initialization: " << e.what() << std::endl;
        return -1;  // Return a negative value to indicate failure
    }

    return 0;  // Return 0 to indicate success
}

#endif // NP_TEST_CPP