import os
from typing import List, Tuple

# @set_module('lightnumpy')
def get_include() -> str:
    """
    Return the directory containing the C and C++ headers for the lightnumpy library.

    This function is intended for extension modules that need to compile
    against lightnumpy's C and C++ API.

    Notes
    -----
    When using ``setuptools``, for example in ``setup.py``::

        import lightnumpy as ln
        ...
        Extension('extension_name', ...
                  include_dirs=ln.[get_include()])
        ...

    Returns
    -------
    str
        Path to the directory containing C and C++ header files.

    Examples
    --------
    >>> import lightnumpy as ln
    >>> ln.get_include()
    '/path/to/lightnumpy/_core/include'  # may vary
    """
    import lightnumpy
    if lightnumpy.show_config is None:
        # running from lightnumpy source directory
        d = os.path.join(os.path.dirname(lightnumpy.__file__), "_core", "include")
    else:
        # using installed lightnumpy core headers
        import lightnumpy._core as core
        # dirname = core.__path__
        d = os.path.join(os.path.dirname(core.__file__), 'include')

    if not os.path.isdir(d):
        raise FileNotFoundError(f"LightNumpy C and C++ headers directory not found: {d}")
    return d