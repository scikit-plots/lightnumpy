// src/hello.cpp
// C++ Source File for printing a customizable message
#ifndef HELLO_CPP  // Include guard to prevent multiple inclusions, If not defined
#define HELLO_CPP

// Standard C++ library header
#include <string>    // For std::string
#include <iostream>  // For std::cout

// Default greeting message definition
#ifndef CPP_MESSAGE                     // 1. Check if CPP_MESSAGE is not defined
#define CPP_MESSAGE "Hello, from C++!"  // 2. If not, define it with the value "Hello, from C++!"
// This block will be ignored if CPP_MESSAGE is already defined.
#endif // CPP_MESSAGE                   // 3. End the condition

// Provide a C-compatible function for external use
// extern "C" disables name mangling, making the function callable from C
// and other languages that expect C-style linkage.
// Without extern "C", name mangling is enabled,
// and the function is only directly usable from C++.
// C++ function declaration with a default message.
// Ensures compatibility with C, Python, or other languages expecting C-style linkage.
extern "C" void printcpp(const std::string& message = CPP_MESSAGE) {
    // Output the message to the console
    std::cout << message << std::endl;
}

#endif // HELLO_CPP