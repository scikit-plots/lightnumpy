// src/py_nc_random.cpp
// C++ Source File for generating random NumPy arrays using pybind11 and optionally NumCpp

#ifndef PY_NC_RANDOM_CPP  // Include guard to prevent multiple inclusions
#define PY_NC_RANDOM_CPP

// Standard C++ library header
#include <stdexcept>             // For runtime error handling
#include <vector>                // Part of the C++ Standard Template Library (STL)

// pybind11 library header
#include <pybind11/pybind11.h>   // For pybind11 functionality
#include <pybind11/numpy.h>      // For handling NumPy arrays
namespace py = pybind11;

#ifndef NUMCPP_NO_INCLUDE
#include <NumCpp.hpp>  // Include NumCpp if available
#endif

/**
 * Generate a random NumPy array.
 * If NumCpp is available, it uses NumCpp to generate the array.
 * Otherwise, it falls back to a pybind11-based implementation.
 *
 * @param rows Number of rows in the array.
 * @param cols Number of columns in the array.
 * @return py::array_t<double> A 2D NumPy array filled with random values.
 */
extern "C" py::array_t<double> random_array(int rows, int cols) {
    // Validate input dimensions
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Rows and columns must be positive integers.");
    }

#ifdef NUMCPP_NO_INCLUDE
    // Use NumCpp to generate a random array
    auto array = nc::random::rand<double>(
        {static_cast<nc::uint32>(rows), static_cast<nc::uint32>(cols)}
    );

    // Prepare the buffer information for the NumPy array
    py::buffer_info buf_info(
        array.data(),                               // Pointer to data
        sizeof(double),                             // Size of one element
        py::format_descriptor<double>::format(),    // NumPy-compatible format
        2,                                          // Number of dimensions
        {static_cast<size_t>(rows), static_cast<size_t>(cols)},  // Shape
        {static_cast<size_t>(cols) * sizeof(double), sizeof(double)}  // Strides
    );

    // Return the NumPy array
    return py::array_t<double>(buf_info);

#else // NUMCPP_NO_INCLUDE
    // Fallback implementation: generate random numbers using pybind11
    std::vector<double> data(rows * cols);
    for (auto& value : data) {
        value = static_cast<double>(rand()) / RAND_MAX;  // Generate random double [0, 1)
    }

    // Prepare the buffer information for the NumPy array
    py::buffer_info buf_info(
        data.data(),                                 // Pointer to data
        sizeof(double),                              // Size of one element
        py::format_descriptor<double>::format(),     // NumPy-compatible format
        2,                                           // Number of dimensions
        {static_cast<size_t>(rows), static_cast<size_t>(cols)},  // Shape
        {static_cast<size_t>(cols) * sizeof(double), sizeof(double)}  // Strides
    );

    // Return the NumPy array
    return py::array_t<double>(buf_info);
#endif // NUMCPP_NO_INCLUDE
}

#endif // PY_NC_RANDOM_CPP