/* -*- c -*- */

/*
 * This is a dummy module designed to ensure that distutils generates
 * the necessary configuration files before building the libraries.
 * 
 * The module doesn't provide any actual functionality but serves as a placeholder
 * for the build process.
 */

#define LNPY_NO_DEPRECATED_API LNPY_API_VERSION  // Define API version for LNPY

// #define NPY_NO_DEPRECATED_API NPY_API_VERSION  // Define API version for NPY
// #define NO_IMPORT_ARRAY  // Prevent NumPy from being imported here

#define PY_SSIZE_T_CLEAN  // Ensures that the Python API uses properly sized ssize_t for Python objects
#include <Python.h>  // Python C API header

// Define the methods for the module (empty in this case)
static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}  // No methods provided
};

// Define the module structure
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  // Module definition structure
    "dummy",                // Module name
    NULL,                   // Module documentation (none)
    -1,                     // Size of per-interpreter state (none here)
    methods,                // Methods for the module (none here)
    NULL,                   // Slot table (none here)
    NULL,                   // Traverse function (none here)
    NULL,                   // Clear function (none here)
    NULL                    // Free function (none here)
};

/* 
 * Initialization function for the dummy module.
 * This function is called by Python when the module is imported or initialized.
 * 
 * @return PyObject* - The created module or NULL on error.
 */
PyMODINIT_FUNC PyInit__dummy(void) {
    PyObject *m;
    
    // Create the module using the module definition
    m = PyModule_Create(&moduledef);
    
    // Check for errors during module creation
    if (!m) {
        // Return NULL on failure to create the module
        return NULL;
    }
    
    // Return the created module
    return m;
}