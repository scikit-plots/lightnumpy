// src/nc_version.cpp
// C++ Source File for retrieving the NumCpp version information

#pragma once  // Ensures the file is included only once by the compiler

#ifndef NC_VERSION_CPP  // Include guard to prevent multiple inclusions
#define NC_VERSION_CPP

// Standard C++ library headers
#include <string>   // For std::string
#include <sstream>  // For std::ostringstream (optional, if needed for other operations)
#include <iostream> // For std::cout (optional, if needed for debugging)

//////////////////////////////////////////////////////////////////////
// NumCpp Header implemented functions by LNPY_USE_NUMCPP
//////////////////////////////////////////////////////////////////////

// Check for the availability of NumCpp headers
#ifdef LNPY_USE_NUMCPP  // If NumCpp is not included (controlled by a macro)
  #include "Version.hpp"  // NumCpp version header containing `nc::VERSION`
  
  // Define NC_VERSION with the actual version or fallback value
  #ifndef NC_VERSION
    #define NC_VERSION nc::VERSION  // Use NumCpp version if available
  #endif // NC_VERSION
#else
  // Fallback: Define NC_VERSION as "Unavailable" if NumCpp is not available
  #define NC_VERSION "Unavailable"
#endif // LNPY_USE_NUMCPP

#ifdef __cplusplus
// Provide C-Compatible Interface (explicitly defined for each type)
// Ensures compatibility with C, Python, or other languages expecting C-style linkage.
extern "C" {
#endif

/**
 * Retrieves the NumCpp version string.
 * This function is marked with `extern "C"` to ensure that it has C-style linkage, 
 * making it compatible with other languages like Python, C, and others that need C-style functions.
 * 
 * @return const char* - The version string of NumCpp, or "Unavailable" if NumCpp headers are not included.
 */
const char* get_numcpp_version() {
  if (std::string(NC_VERSION) == "Unavailable") {
    std::cerr << "Warning: NumCpp version is unavailable. Make sure the NumCpp library is included." << std::endl;
  }
  return NC_VERSION;  // Return the version string (either nc::VERSION or "Unavailable")
}

/**
 * Alias function for `get_numcpp_version`.
 * This function provides an alternative name for retrieving the version.
 * Useful for maintaining consistency in naming conventions across different parts of the code.
 *
 * @return const char* - The version string of NumCpp.
 */
const char* nc_version() {
  return get_numcpp_version();  // Delegate to the original function `get_numcpp_version`
}

#ifdef __cplusplus
}  // End of `extern "C"` block
#endif

#endif // NC_VERSION_CPP