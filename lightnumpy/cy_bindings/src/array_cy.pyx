# Template: Implement array-related operations in Cython.

# Example Cython code using external C++ function
cimport cython_helpers

def print_message(message: str):
    cython_helpers.printcpp(message.encode('utf-8'))