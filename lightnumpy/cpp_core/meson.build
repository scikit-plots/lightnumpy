######################################################################
## Submodule NumCpp
######################################################################

if not fs.exists('NumCpp/README.md')
  error('Missing the `NumCpp` submodule! ' +
  'Run `git submodule update --init` to fix this.')
endif

######################################################################
## Compiler Flags
######################################################################

_global_numcpp_args = []

# Initialize an empty list for compiler flags
_global_numcpp_args += [
  # to ensure that the generated code can be executed from any address in memory,
  # which is necessary for shared libraries.
  # '-fPIC',
  '-DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE', # Include PyBind11 Python interface helper functions
]

# Optional Boost Dependency
# dpkg -s libboost-dev | grep 'Version'
if get_option('numcpp_no_use_boost')
  # Disable all NumCpp features that require Boost
  _global_numcpp_args += ['-DNUMCPP_NO_USE_BOOST']
else
  _global_numcpp_args += [
    '-DNUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE', # Include Boost Python interface helper functions
  ]
  # https://dpilger26.github.io/NumCpp/doxygen/html/index.html
  dep_boost = dependency('boost', version: '>=1.73.0',
                         required: not get_option('numcpp_no_use_boost'))
  dependencies_list += [dep_boost]
  message('Dependency: Boost dependency added...')
endif

# Optional TBB Dependency for Multithreading
if get_option('numcpp_use_multithread')
  # Enable multithreading with STL parallel execution policies
  _global_numcpp_args += [
    '-DNUMCPP_USE_MULTITHREAD',               # Enable multithreading with STL parallel execution policies
  ]
  dep_tbb = dependency('tbb', required: get_option('numcpp_use_multithread'))
  dependencies_list += [dep_tbb]
  message('Dependency: Multithreading TBB dependency added...')
endif

_global_numcpp_args += _c_flags

######################################################################
## include the Headers directories
######################################################################

# include directories for Specific Subfolders
_numcpp_inc_dir = [
  'src',
  'include',
  'NumCpp/include/NumCpp/Core/Internal',
  'NumCpp/include',
  # 'NumCpp/include/NumCpp',
  # 'NumCpp/include/NumCpp/Coordinates',
  # 'NumCpp/include/NumCpp/Coordinates/ReferenceFrames',
  # 'NumCpp/include/NumCpp/Coordinates/Transforms',
  # 'NumCpp/include/NumCpp/Core',
  # 'NumCpp/include/NumCpp/Core/Internal',
  # 'NumCpp/include/NumCpp/DateTime',
  # 'NumCpp/include/NumCpp/Filter',
  # 'NumCpp/include/NumCpp/Filter/Boundaries',
  # 'NumCpp/include/NumCpp/Filter/Boundaries/Boundaries1d',
  # 'NumCpp/include/NumCpp/Filter/Boundaries/Boundaries2d',
  # 'NumCpp/include/NumCpp/Filter/Filters',
  # 'NumCpp/include/NumCpp/Filter/Filters/Filters1d',
  # 'NumCpp/include/NumCpp/Filter/Filters/Filters2d',
  # 'NumCpp/include/NumCpp/Functions',
  # 'NumCpp/include/NumCpp/ImageProcessing',
  # 'NumCpp/include/NumCpp/Integrate',
  # 'NumCpp/include/NumCpp/Linalg',
  # 'NumCpp/include/NumCpp/Linalg/svd',
  # 'NumCpp/include/NumCpp/Logging',
  # 'NumCpp/include/NumCpp/NdArray',
  # 'NumCpp/include/NumCpp/Polynomial',
  # 'NumCpp/include/NumCpp/PythonInterface',
  # 'NumCpp/include/NumCpp/Random',
  # 'NumCpp/include/NumCpp/Roots',
  # 'NumCpp/include/NumCpp/Rotations',
  # 'NumCpp/include/NumCpp/Special',
  # 'NumCpp/include/NumCpp/Utils',
  # 'NumCpp/include/NumCpp/Vector',
]

######################################################################
## Include Header-only NumCpp
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_numcpp = include_directories(_numcpp_inc_dir)

# Optionally add a message to confirm the installation
message(
  '\nNumCpp Header compatibility files defined successfully: ' +
  '@0@'.format('lightnumpy/cpp_core/NumCpp')
)

######################################################################
## NumCpp is a header-only library, which means it does not have any precompiled binary
## (like .so, .dll, or .a). 
######################################################################

# # Static library with C++ source file implementing bindings
# _st_lib_numcpp = static_library('_numcpp_lib_stat', [],
#   include_directories: inc_dir_numcpp,
#   dependencies: [],
#   link_with: [],
#   link_args: [],
#   c_args: _global_numcpp_args,
#   cpp_args: _global_numcpp_args,
#   gnu_symbol_visibility: 'inlineshidden',
#   # install: true,
#   # install_dir: 'lightnumpy/cpp_core',
# )
# # Shared (dynamic) library with C++ source file implementing bindings
# _dyn_lib_numcpp = library('_dyn_lib_numcpp', [],
#   include_directories: inc_dir_numcpp,
#   dependencies: [],
#   # link_with: [_st_lib_numcpp],  # Link with the static library
#   # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'hidden',
#   gnu_symbol_visibility: 'inlineshidden',
#   install: true,
#   install_dir: 'lightnumpy/cpp_core',
# )

# Get the NumCpp Dependencies with/without library
dep_numcpp = declare_dependency(
  compile_args: _global_numcpp_args,
  dependencies: [],
  include_directories: inc_dir_numcpp,
  # link_with: [_st_lib_numcpp],  # Link with the static/shared library
  # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
)
# Optionally add a message to confirm the installation
message(
  '\nNumCpp Header dependency defined successfully: ' +
  '@0@ '.format('dep_numcpp')
)