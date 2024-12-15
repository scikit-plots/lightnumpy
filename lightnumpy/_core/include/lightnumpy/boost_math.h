#pragma once
// Notes:
// - Boost Math includes many advanced mathematical functions that are not available in the standard C++ library, such as Bessel functions (`boost::math::cyl_bessel_j()`), Gamma functions, and more.
// - To use Boost Math, make sure you've installed the Boost C++ libraries and included the relevant headers.
// - The functions are organized into namespaces like `boost::math` and are well-documented. For example, you can use `boost::math::pi` to get the value of π.
// - Boost provides high-precision and efficient implementations, especially for complex mathematical tasks such as statistical distributions and interpolation.

// Optional Boost Math C++ Libraries
// Boost Math provides a wide range of advanced mathematical functions that are optimized for performance and precision.
#ifdef BOOST_MATH_INCLUDED  // if defined(BOOST_MATH_INCLUDED)
    #include <boost/math/special_functions.hpp>
    #include <boost/math/constants/constants.hpp>
#else
    // ifndef BOOST_MATH_INCLUDED  // if !defined(BOOST_MATH_INCLUDED)
    // Boost Math is optional, so no error is raised.
    // If you need Boost Math, install it and ensure it's available.
#endif
