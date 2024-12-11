// src/nc_version.cpp
// C++ Source File for retrieving the NumCpp version information
#ifndef NC_VERSION_CPP  // Include guard to prevent multiple inclusions, If not defined
#define NC_VERSION_CPP

// Standard C++ library header
#include <string>           // For std::string
// #include <iostream>

#ifndef NUMCPP_NO_INCLUDE
// Include NumCpp if available
#include "Version.hpp"  // NumCpp version header nc::VERSION

// Ensure NUMCPP_VERSION is defined if not already specified
#ifndef NUMCPP_VERSION
#define NUMCPP_VERSION nc::VERSION
#endif // NUMCPP_VERSION

#else // NUMCPP_NO_INCLUDE
// Fallback: Define NUMCPP_VERSION as "Unavailable" if NumCpp is not available
#ifndef NUMCPP_VERSION
#define NUMCPP_VERSION "Unavailable"
#endif // NUMCPP_VERSION
#endif // NUMCPP_NO_INCLUDE

/**
 * Get the version of NumCpp.
 * Provides compatibility with other languages via extern "C".
 * Ensures compatibility with C, Python, or other languages expecting C-style linkage.
 *
 * @return const char* - The version string of NumCpp.
 */
extern "C" const char* numcpp_version() {
  // Print the version
  // std::cout << nc::VERSION << std::endl;

  // Return the version string directly
  // nc::VERSION is a const char[], safe to return directly
  return NUMCPP_VERSION;
}


/**
 * Alias function for numcpp_version.
 * Useful for maintaining consistent naming conventions.
 *
 * @return const char* - The version string of NumCpp.
 */
extern "C" const char* nc_version() {
  // Print the version
  // std::cout << "Version: " << nc::VERSION << std::endl;
  // std::cout << "Version: " << numcpp_version() << std::endl;

    // Delegate to the original function numcpp_version
    return numcpp_version();
}

#endif // NC_VERSION_CPP