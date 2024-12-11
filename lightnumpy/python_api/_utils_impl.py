import os
from typing import List, Tuple


# @set_module('lightnumpy')
def get_c_include() -> str:
    """
    Return the directory containing the C headers for the lightnumpy library.

    This function is intended for extension modules that need to compile
    against lightnumpy's C API.

    Notes
    -----
    When using ``setuptools``, for example in ``setup.py``::

        import lightnumpy as ln
        ...
        Extension('extension_name', ...
                  include_dirs=[ln.get_c_include()])
        ...

    Returns
    -------
    str
        Path to the directory containing C header files.

    Examples
    --------
    >>> import lightnumpy as ln
    >>> ln.get_c_include()
    '/path/to/lightnumpy/_c_core/include'  # may vary
    """
    import lightnumpy
    if lightnumpy.show_config is None:
        # running from lightnumpy source directory
        d = os.path.join(os.path.dirname(lightnumpy.__file__), "_c_core", "include")
    else:
        # using installed lightnumpy core headers
        import lightnumpy._c_core as _core
        d = os.path.join(os.path.dirname(_core.__file__), 'include')

    if not os.path.isdir(d):
        raise FileNotFoundError(f"LightNumpy C headers directory not found: {d}")
    return d

# @set_module('lightnumpy')
def get_cpp_include() -> str:
    """
    Return the directory containing the C++ headers for the lightnumpy library.

    This function is intended for extension modules that need to compile
    against lightnumpy's C++ API.

    Notes
    -----
    When using ``setuptools``, for example in ``setup.py``::

        import lightnumpy as ln
        ...
        Extension('extension_name', ...
                  include_dirs=[ln.get_cpp_include()])
        ...

    Returns
    -------
    str
        Path to the directory containing C++ header files.

    Examples
    --------
    >>> import lightnumpy as ln
    >>> ln.get_cpp_include()
    '/path/to/lightnumpy/_cpp_core/include'  # may vary
    """
    import lightnumpy
    if lightnumpy.show_config is None:
        # running from lightnumpy source directory
        d = os.path.join(os.path.dirname(lightnumpy.__file__), "_cpp_core", "include")
    else:
        # using installed lightnumpy core headers
        import lightnumpy._cpp_core as _core
        d = os.path.join(os.path.dirname(_core.__file__), 'include')

    if not os.path.isdir(d):
        raise FileNotFoundError(f"LightNumpy C++ headers directory not found: {d}")
    return d

# @set_module('lightnumpy')
def get_include() -> Tuple[str]:
    """
    Return a tuple of directories containing the C and C++ headers for the lightnumpy library.

    Extension modules that need to compile against lightnumpy may use this
    function to locate the appropriate include directories.

    Notes
    -----
    When using ``setuptools``, for example in ``setup.py``::

        import lightnumpy as ln
        ...
        Extension('extension_name', ...
                  include_dirs=ln.get_include())
        ...

    Returns
    -------
    tuple of str
        Paths to the C and C++ headers.

    Examples
    --------
    >>> import lightnumpy as ln
    >>> ln.get_include()
    ('/path/to/lightnumpy/_c_core/include', '/path/to/lightnumpy/_cpp_core/include')  # may vary
    """
    return get_c_include(), get_cpp_include()