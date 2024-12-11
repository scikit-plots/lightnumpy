// src/py_math.cpp
// C++ Source File for Python bindings to perform mathematical operations
#ifndef PY_MATH_CPP  // Include guard to prevent multiple inclusions, If not defined
#define PY_MATH_CPP

// Standard C++ library header
#include <stdexcept>            // For runtime error handling

// pybind11 library header
#include <pybind11/pybind11.h>  // For pybind11 support
#include <pybind11/numpy.h>     // For handling NumPy arrays
namespace py = pybind11;

// C++ Function to calculate the sum of squares of elements in a NumPy array
extern "C" double sum_of_squares(const py::array_t<double>& input_array) {
    // Obtain buffer information from the input NumPy array
    py::buffer_info buf_info = input_array.request();
    double* data_ptr = static_cast<double*>(buf_info.ptr);  // Pointer to the data
    size_t array_size = buf_info.size;                      // Size of the array

    // Handle the case of an empty input array
    if (array_size == 0) {
        throw std::runtime_error("Input array is empty. Cannot compute sum of squares.");
    }

    // Compute the sum of squares
    double sum = 0.0;
    for (size_t i = 0; i < array_size; ++i) {
        sum += data_ptr[i] * data_ptr[i];  // Square each element and add to the sum
    }

    return sum;  // Return the computed sum
}

#endif // PY_MATH_CPP