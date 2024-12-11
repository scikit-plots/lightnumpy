// src/hello.c
// C Header File for printing a customizable message
#ifndef HELLO_C  // Include guard to prevent multiple inclusions, If not defined
#define HELLO_C

// Standard C library header
#include <stdio.h>  // Standard I/O library for printf
#include <string.h> // For string operations if needed

// Default greeting message definition
#ifndef C_MESSAGE                     // 1. Check if C_MESSAGE is not defined
#define C_MESSAGE "Hello, from C!"    // 2. If not, define it with the value "Hello, from C!"
// This block will be ignored if C_MESSAGE is already defined.
#endif // C_MESSAGE                   // 3. End the condition

// Function declaration for printing a message
#ifdef __cplusplus
// Provide a C-compatible function for external use
// extern "C" disables name mangling, making the function callable from C
// and other languages that expect C-style linkage.
// Without extern "C", name mangling is enabled,
// and the function is only directly usable from C++.
// C++ function declaration with a default message.
// Ensures compatibility with C, Python, or other languages expecting C-style linkage.
extern "C" {
#endif

// void print(const char* message = C_MESSAGE);
void printc(const char* message = C_MESSAGE) {
    if (message == NULL || strlen(message) == 0) {
        message = C_MESSAGE;  // Use the default message if none is provided
    }
    // Print a message
    printf("%s\n", message);
}

#ifdef __cplusplus
}
#endif

#endif // HELLO_C