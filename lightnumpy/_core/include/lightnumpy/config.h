// include/config.h
// Unified C and C++ Header File

#pragma once  // Ensures the file is included only once by the compiler

#ifndef CONFIG_H
#define CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

// Common configurations
#define APP_VERSION "1.0.0"
#define DEBUG_MODE 1

// Function declarations
// void initialize_app();

// const char* get_app_version();
const char* get_app_version() {  // config.c
    return APP_VERSION;
}

#ifdef __cplusplus
} // extern "C"

// C++-specific configurations or features
#include <iostream>
inline void print_version() {
    std::cout << "App version: " << APP_VERSION << std::endl;
}
#endif

#include <string>
inline std::string get_app_version_cpp() {
    return std::string(get_app_version());
}
#endif

#endif // CONFIG_H