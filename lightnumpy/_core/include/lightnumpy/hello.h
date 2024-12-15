// include/hello.h
// Unified C and C++ Header File for printing customizable messages and conversions

#pragma once  // Ensures the file is included only once by the compiler

#ifndef HELLO_H  // Traditional include guard for compatibility
#define HELLO_H

#ifndef MIN_BUFFER_SIZE
#define MIN_BUFFER_SIZE 64   // Define minimum buffer size
#endif

#ifndef MAX_BUFFER_SIZE
#define MAX_BUFFER_SIZE 1024 // Define maximum buffer size
#endif

// Include standard headers
#ifdef __cplusplus
    // C++ specific headers
    #include <string>    // For std::string (C++ only)
    #include <cstring>
    #include <sstream>   // For std::ostringstream (C++ only)
    #include <iostream>  // For std::cout (C++ only)
    #include <cmath>     // For math functions (C++ version of math.h)
    #include <vector>    // For std::vector (C++ only)
#else
    // C specific headers
    #include <stdio.h>   // For printf and sprintf (C only)
    #include <string.h>  // For string operations (C only)
    #include <stdlib.h>  // For memory allocation and process control (C only)
    #include <math.h>    // For math functions (C only)
#endif // End of standard headers inclusion

// Default greeting message definition
#ifndef DEFAULT_MESSAGE
#ifdef __cplusplus
#define DEFAULT_MESSAGE "Hello, from C++!"
#else
#define DEFAULT_MESSAGE "Hello, from C!"
#endif
#endif  // Default message

#ifdef __cplusplus  // C++ code begins here
// Provide C-Compatible Interface (explicitly defined for each type)
// extern "C" disables name mangling, making the function callable from C
// and other languages that expect C-style linkage.
// Without extern "C", name mangling is enabled,
// and the function is only directly usable from C++.
// C++ function declaration with a default message.
// Ensures compatibility with C, Python, or other languages expecting C-style linkage.
extern "C" {

//////////////////////////////////////////////////////////////////////
// Begin of C-style Code (for C functions)
// C functions are automatically compatible with C and C++
//////////////////////////////////////////////////////////////////////

/**
 * @brief Prints the given message to the console. For C, only C-style strings are supported.
 *
 * @param message A C-style string to print. Defaults to DEFAULT_MESSAGE if NULL or empty.
 */
// Function declaration for C (using const char* for C-style string)
// void to_print(const char* message = DEFAULT_MESSAGE);  // Normally Header file function definition
void c_char_to_print(const char* message = DEFAULT_MESSAGE) {  // We adding here hello.c
    // if (message == NULL || message[0] == '\0') {
    if (message == NULL || strlen(message) == 0) {
        message = DEFAULT_MESSAGE;  // Use the default message if none is provided
    }
    printf("%s\n", message);  // Print the message
}

/**
 * @brief Converts an integer to a string in C.
 *
 * @param value The integer value to convert.
 * @param buffer The buffer to store the resulting string. Must be large enough to hold the result.
 */
// Function to convert integer to string
// void to_string_int(int value, char* buffer);
void c_int_to_string(int value, char** buffer) {
    // Calculate required size for integer conversion
    int size_needed = snprintf(NULL, 0, "%d", value) + 1; // +1 for null-termination

    // Ensure buffer size is at least the minimum size
    int buffer_size = (size_needed < MIN_BUFFER_SIZE) ? MIN_BUFFER_SIZE : size_needed;

    // Allocate memory for the buffer
    *buffer = (char*)malloc(buffer_size);
    if (*buffer == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed for integer conversion.\n");
        exit(EXIT_FAILURE);  // Exit the program with failure code
    }

    // Format the integer into the buffer
    snprintf(*buffer, buffer_size, "%d", value);
}

/**
 * @brief Converts a float to a string in C with a specified number of decimal places.
 * Defaults to 2 decimal places if the user doesn't specify.
 *
 * @param value The float value to convert.
 * @param buffer The buffer to store the resulting string. Must be large enough to hold the result.
 * @param decimal_places The number of decimal places to include. Defaults to 2.
 */
// Function to convert float to string with specified decimal places
// void to_string_float(float value, char* buffer, int decimal_places = 2);
void c_float_to_string(float value, char** buffer, int decimal_places = 2) {
    if (decimal_places <= 0) {
        decimal_places = 2;  // Default to 2 decimal places if not provided
    }

    // Calculate required size for float conversion
    int size_needed = snprintf(NULL, 0, "%.*f", decimal_places, value) + 1; // +1 for null-termination

    // Ensure buffer size is within min/max range
    int buffer_size = (size_needed < MIN_BUFFER_SIZE) ? MIN_BUFFER_SIZE : 
                      (size_needed > MAX_BUFFER_SIZE ? MAX_BUFFER_SIZE : size_needed);

    // Allocate memory for the buffer
    *buffer = (char*)malloc(buffer_size);
    if (*buffer == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed for float conversion.\n");
        exit(EXIT_FAILURE);  // Exit the program with failure code
    }

    // Format the float into the buffer
    snprintf(*buffer, buffer_size, "%.*f", decimal_places, value);
}

/**
 * @brief Converts a C-style string to a string or DEFAULT_MESSAGE if null/empty.
 *
 * @param value The C-style string to convert.
 * @param buffer The buffer to store the resulting string. Must be large enough to hold DEFAULT_MESSAGE.
 */
// Function to convert C-style string to string
// void to_string_cstr(const char* value, char* buffer);
void c_char_to_string(const char* value, char** buffer) {
    if (value == NULL || value[0] == '\0') {
        // Allocate memory for default message
        *buffer = (char*)malloc(strlen(DEFAULT_MESSAGE) + 1);
        if (*buffer == NULL) {
            // Handle memory allocation failure
            fprintf(stderr, "Memory allocation failed for default message.\n");
            exit(EXIT_FAILURE);  // Exit the program with failure code
        }
        strcpy(*buffer, DEFAULT_MESSAGE);  // Copy default message to buffer
    } else {
        // Allocate memory for the provided string
        int size_needed = strlen(value) + 1;  // +1 for null-termination
        *buffer = (char*)malloc(size_needed);
        if (*buffer == NULL) {
            // Handle memory allocation failure
            fprintf(stderr, "Memory allocation failed for string conversion.\n");
            exit(EXIT_FAILURE);  // Exit the program with failure code
        }
        strcpy(*buffer, value);  // Copy the string to the buffer
    }
}

//////////////////////////////////////////////////////////////////////
// Begin of C++-style Code (for C++ functions)
//////////////////////////////////////////////////////////////////////

/**
 * @brief Prints the given message to the console. Supports both C-style and C++ strings.
 *
 * @param message A string to print. Accepts `const char*` or `std::string`.
 */
// C++ version of to_print that accepts std::string (C++ only)
inline void cpp_str_to_print(const std::string& message = DEFAULT_MESSAGE) {
    std::cout << (message.empty() ? DEFAULT_MESSAGE : message) << std::endl;  // Output the message to the console
}
inline void cpp_char_to_print(const char* message = DEFAULT_MESSAGE) {
    if (!message || strlen(message) == 0) {
        std::cout << DEFAULT_MESSAGE << std::endl;  // Output the message to the console
    } else {
        std::cout << message << std::endl;  // Output the message to the console
    }
}

// Specialization for `std::string`
inline std::string cpp_str_to_string(const std::string& value) {
    // Return the input string, but use DEFAULT_MESSAGE if it's empty
    return value.empty() ? DEFAULT_MESSAGE : value;
}
// Specialization for `const char*`
inline std::string cpp_char_to_string(const char* value) {
    // Return default message for null or empty C strings
    // if (!value || strlen(value) == 0) {return DEFAULT_MESSAGE;}
    // if (value == nullptr || strlen(value) == 0) {return DEFAULT_MESSAGE;}
    // return std::string(value);  // Return the input string as is
  
    // Return default message for null or empty C strings
    return (value && strlen(value) > 0) ? value : DEFAULT_MESSAGE;
}

}
#endif  // End of C++-style Code
//////////////////////////////////////////////////////////////////////

#endif // HELLO_H