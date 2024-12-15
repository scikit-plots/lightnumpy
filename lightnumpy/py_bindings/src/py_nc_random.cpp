// src/py_nc_random.cpp
// C++ Source File for generating random NumPy arrays using pybind11 and optionally NumCpp

#ifndef PY_NC_RANDOM_CPP  // Include guard to prevent multiple inclusions
#define PY_NC_RANDOM_CPP

// Standard C++ library headers
#include <stdexcept>             // For runtime error handling
#include <vector>                // For STL vector

//////////////////////////////////////////////////////////////////////
// pybind11 Header implemented functions by LNPY_USE_NUMCPP
//////////////////////////////////////////////////////////////////////

// pybind11 library headers
#include <pybind11/pybind11.h>   // For pybind11 functionality
#include <pybind11/numpy.h>      // For handling NumPy arrays
namespace py = pybind11;

//////////////////////////////////////////////////////////////////////
// NumCpp Header implemented functions by LNPY_USE_NUMCPP
//////////////////////////////////////////////////////////////////////

#ifdef LNPY_USE_NUMCPP
#include <NumCpp.hpp>            // Include NumCpp if available
#endif

#ifdef __cplusplus
// Ensures compatibility with C, Python, or other languages expecting C-style linkage.
extern "C" {
#endif

#ifdef LNPY_USE_NUMCPP
/**
 * @brief Helper function to generate random array using NumCpp.
 * 
 * @param rows Number of rows in the array.
 * @param cols Number of columns in the array.
 * @return py::array_t<double> A 2D NumPy array filled with random values.
 * @throws std::runtime_error if NumCpp generation fails.
 */
py::array_t<double> generate_random_array_with_numcpp(int rows, int cols) {
    try {
        // Use NumCpp to generate a random array
        auto array = nc::random::rand<double>({static_cast<nc::uint32>(rows), static_cast<nc::uint32>(cols)});

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
    } catch (const std::exception& e) {
        throw std::runtime_error("Error in NumCpp array generation: " + std::string(e.what()));
    }
}
#endif

/**
 * @brief Helper function to generate random array using pybind11 (fallback).
 * 
 * @param rows Number of rows in the array.
 * @param cols Number of columns in the array.
 * @return py::array_t<double> A 2D NumPy array filled with random values.
 * @throws std::runtime_error if pybind11 generation fails.
 */
py::array_t<double> generate_random_array_with_pybind11(int rows, int cols) {
    try {
        // Fallback: generate random numbers using pybind11 and C++ standard library
        std::vector<double> array(rows * cols);
        for (auto& value : array) {
            value = static_cast<double>(rand()) / RAND_MAX;  // Generate random double [0, 1)
        }

        // Prepare the buffer information for the NumPy array
        py::buffer_info buf_info(
            array.data(),                                 // Pointer to data
            sizeof(double),                              // Size of one element
            py::format_descriptor<double>::format(),     // NumPy-compatible format
            2,                                           // Number of dimensions
            {static_cast<size_t>(rows), static_cast<size_t>(cols)},  // Shape
            {static_cast<size_t>(cols) * sizeof(double), sizeof(double)}  // Strides
        );

        // Return the NumPy array
        return py::array_t<double>(buf_info);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error in pybind11 random array generation: " + std::string(e.what()));
    }
}

/**
 * @brief Generate a random NumPy array.
 * 
 * This function generates a 2D NumPy array with random values. It first checks if
 * NumCpp is available. If it is, it uses NumCpp to generate the array. Otherwise,
 * it falls back to a pybind11-based implementation that uses the C++ Standard Library.
 * 
 * @param rows Number of rows in the array (must be positive).
 * @param cols Number of columns in the array (must be positive).
 * @return py::array_t<double> A 2D NumPy array filled with random values.
 * @throws std::invalid_argument if rows or columns are non-positive.
 * @throws std::runtime_error if NumCpp is unavailable and pybind11-based fallback fails.
 */
py::array_t<double> random_array(int rows, int cols) {
    // Validate input dimensions
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Rows and columns must be positive integers.");
    }
    // Check if NumCpp is available and use it if possible
    #ifdef LNPY_USE_NUMCPP
        // NumCpp is available, use it to generate the random array
        try {
            return generate_random_array_with_numcpp(rows, cols);
        } catch (const std::exception& e) {
            throw std::runtime_error("NumCpp random array generation failed: " + std::string(e.what()));
        }
    #else
        // Fallback implementation: generate random numbers using pybind11
        try {
            return generate_random_array_with_pybind11(rows, cols);
        } catch (const std::exception& e) {
            throw std::runtime_error("Fallback pybind11 random array generation failed: " + std::string(e.what()));
        }
    #endif
}

#ifdef __cplusplus
}  // End of extern "C" block
#endif

#endif // PY_NC_RANDOM_CPP