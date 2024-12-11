# Template: Declare Cython-exposed C/C++ functions for use in .pyx files.

# Declare external C++ functions
cdef extern from "hello.cpp":
    void printcpp(const char* message)
