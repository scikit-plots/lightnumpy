// src/py_math.cpp
// C++ Source File for Python bindings to perform mathematical operations

#ifndef PY_MATH_CPP  // Include guard to prevent multiple inclusions
#define PY_MATH_CPP

// Standard C++ library headers
#include <stdexcept>            // For runtime error handling

//////////////////////////////////////////////////////////////////////
// pybind11 Header implemented functions
//////////////////////////////////////////////////////////////////////

// pybind11 library headers
#include <pybind11/pybind11.h>  // For pybind11 support
#include <pybind11/numpy.h>     // For handling NumPy arrays

namespace py = pybind11;

#ifdef __cplusplus
// Ensures compatibility with C, Python, or other languages expecting C-style linkage.
extern "C" {
#endif

/**
 * @brief Computes the sum of squares of elements in a NumPy array.
 * 
 * This function calculates the sum of squares of the elements in the provided 
 * NumPy array (input_array). It handles cases where the array is empty and 
 * throws an exception with a detailed message if an error occurs.
 * 
 * @param input_array The NumPy array (array_t<double>) containing the elements.
 * @return double The sum of squares of the elements in the input array.
 * @throws std::invalid_argument if the input array is empty.
 */
double sum_of_squares(const py::array_t<double>& input_array) {
    try {
        // Obtain buffer information from the input NumPy array
        py::buffer_info buf_info = input_array.request();
        double* data_ptr = static_cast<double*>(buf_info.ptr);  // Pointer to the data
        size_t array_size = buf_info.size;                      // Size of the array

        // Handle the case of an empty input array
        if (array_size == 0) {
            throw std::invalid_argument("Input array is empty. Cannot compute sum of squares.");
        }

        // Compute the sum of squares
        double sum = 0.0;
        for (size_t i = 0; i < array_size; ++i) {
            sum += data_ptr[i] * data_ptr[i];  // Square each element and add to the sum
        }

        return sum;  // Return the computed sum
    } catch (const std::exception& e) {
        throw std::runtime_error("Error in sum_of_squares: " + std::string(e.what()));
    }
}

#ifdef __cplusplus
}  // End of `extern "C"` block
#endif

#endif // PY_MATH_CPP