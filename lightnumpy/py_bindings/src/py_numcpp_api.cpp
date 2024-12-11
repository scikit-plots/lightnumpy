// C++ Source File
// Pybind11 implementation write C++ directly
// Pybind11 focuses on bridging C++ with Python directly.
// .cpp Files: Pybind11 uses standard C++ source files.
// You write your binding code directly in C++ using the Pybind11 API
// to expose C++ functions and classes to Python.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // For numpy array bindings
#include <pybind11/stl.h>   // For STL bindings
namespace py = pybind11;

#include "hello.cpp"        // Include header or function prototypes if needed
#include "nc_version.cpp"   // Include for version
#include "py_math.cpp"      // Include header or function prototypes if needed

#ifndef NUMCPP_NO_INCLUDE
#include "py_nc_random.cpp"    // Include header or function prototypes if needed

void numcpp_function(pybind11::module_ &m) {
  // Add bindings here
  // AdD More numcpp feature  
  // Define module functions using a Function Pointer and a docstring
  m.def("py_random_array",
  &random_array,
  R"(\
Create a random NumCpp array.

Parameters
----------
arg0 : int
    Row.
    
arg1 : int
    Col.

Returns
-------
numpy.array
    2D array-like, shape (arg0, arg1)

Examples
--------
.. jupyter-execute::

    >>> import numpy as np; np.random.seed(0)
    >>> from lightnumpy.py_bindings import py_numcpp_api as lp
    >>> arr = lp.py_random_array(1, 2)
    >>> arr
)"
  );
}
#endif  // NUMCPP_NO_INCLUDE


// Expose the functions to Python Module using Pybind11
// In practice, implementation and binding code will generally be located in separate files.
// https://pybind11.readthedocs.io/en/stable/reference.html#c.PYBIND11_MODULE
PYBIND11_MODULE(py_numcpp_api, m) {
  m.doc() = 
    R"(\
NumCpp API Python module that uses Pybind11 C/C++ for numerical computations.
Created by Pybind11.
)";
  // optional module docstring  
  // Add bindings here
  // Define module functions using a Lambda Function and a docstring
  m.def("py_print",
    [](std::string message =
      "Hello, from Pybind11 C++!") { printcpp(message); },
    py::arg("message") = "Hello, from Pybind11 C++!",
    R"(\
Prints a Unicode message.

Parameters
----------
message : str, optional, default='Hello, from Pybind11 C++!'
    Prints a Unicode message.

Returns
-------
None
    Prints a Unicode message.

Examples
--------
.. jupyter-execute::

    >>> from lightnumpy.py_bindings import py_numcpp_api as lp
    >>> lp.py_print()
)"
  );

  // Expose the VERSION constant to Python as a function that returns it
  // Return the version string defined in Version.hpp
  m.def("nc_version", []() { 
    // Example: Return a NumPy-like array or another complex object
    return numcpp_version(); 
  },
  R"(\
Get the NumCpp header library version.

Returns
-------
str
    NumCpp header library version.

Examples
--------
.. jupyter-execute::

    >>> from lightnumpy.py_bindings import py_numcpp_api as lp
    >>> lp.nc_version()
)"
  );

  // Define module functions using a Function Pointer and a docstring
  m.def("py_sum_of_squares",
    &sum_of_squares,
    R"(\
Calculate the sum of squares.

Parameters
----------
arg0 : array-like

Returns
-------
float
    Sum of squares.

Examples
--------
.. jupyter-execute::

    >>> import numpy as np; np.random.seed(0)
    >>> from lightnumpy.py_bindings import py_numcpp_api as lp
    >>> arr = np.array([1,2])
    >>> lp.py_sum_of_squares(arr)
)"
  );

  // Conditionally add function or feature based on NUMCPP_NO_INCLUDE
#ifndef NUMCPP_NO_INCLUDE
  numcpp_function(m);  // This will add the function when numcpp is enabled
#endif
}