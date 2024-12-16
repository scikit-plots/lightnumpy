######################################################################
## Define dependencies and Compiler Flags for NumCpp
######################################################################

# Define args and dependencies for NumCpp
_global_numcpp_args = []
_dep_list_numcpp = []

if get_option('lnpy_use_numcpp').enabled()
  # Initialize an empty list for compiler flags
  _global_numcpp_args += [
    # The -D flag defines a macro that is globally available during compilation.
    # The -U flag removes a macro definition during compilation, effectively ignoring it.
    '-DLNPY_USE_NUMCPP',  # Enable all NumCpp features
    '-DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE', # Include PyBind11 Python interface helper functions
    # The -f option in compilers like gcc or g++ is used to enable
    # or disable specific compiler features or behaviors.
    # to ensure that the generated code can be executed from any address in memory,
    # which is necessary for shared libraries.
    # '-fPIC',                  # Generate position-independent code
    # -finline-functions        # Enable function inlining
    # -fno-inline-functions     # Disable function inlining
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
    _dep_list_numcpp += [dep_boost]
    message('Dependency: Boost dependency added...')
  endif
  # Optional TBB Dependency for Multithreading
  if get_option('numcpp_use_multithread')
    # Enable multithreading with STL parallel execution policies
    _global_numcpp_args += [
      '-DNUMCPP_USE_MULTITHREAD',               # Enable multithreading with STL parallel execution policies
    ]
    dep_tbb = dependency('tbb', required: get_option('numcpp_use_multithread'))
    _dep_list_numcpp += [dep_tbb]
    message('Dependency: Multithreading TBB dependency added...')
  endif
else
  _global_numcpp_args += [
    '-U LNPY_USE_NUMCPP',  # Disable all NumCpp features
  ]
endif

cython_cpp_flags += _global_numcpp_args

######################################################################
## Include the Headers directories for _core, NumCpp
######################################################################

# include directories for Specific Subfolders
_numcpp_inc_dir = []
if get_option('lnpy_use_numcpp').enabled()
  _numcpp_inc_dir = [
    '../../third_party/NumCpp/include',
    '../../third_party/NumCpp/include/NumCpp/Core/Internal', # version
  ]
endif

_core_inc_dir = [
  'src',
  'include',
] + _numcpp_inc_dir

######################################################################
## Define include_directories for _core, NumCpp
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_core = include_directories(
  _core_inc_dir
)
# Optionally add a message to confirm the installation
message(
  '\nCpp Core Header compatibility files defined successfully: ' +
  '@0@ '.format('lightnumpy/_core/include') +
  'with NumCpp: @0@'.format(get_option('lnpy_use_numcpp').enabled())
)

######################################################################
## NumCpp is a header-only library, which means it does not have any precompiled binary
## (like .so, .dll, or .a). 
######################################################################

# # Static library with C++ source file implementing bindings
# _st_lib_core = static_library('_st_lib_core', [],
#   include_directories: inc_dir_core,
#   dependencies: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'inlineshidden',
#   link_with: [],
#   link_args: [],
#   # install: true,
#   # install_dir: 'lightnumpy/_core',
# )
# # Shared (dynamic) library with C++ source file implementing bindings
# _dyn_lib_core = library('_dyn_lib_core', [],
#   include_directories: inc_dir_core,
#   dependencies: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'hidden',
#   gnu_symbol_visibility: 'inlineshidden',
#   # link_with: [_st_lib_core],  # Link with the static library
#   # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
#   install: true,
#   install_dir: 'lightnumpy/_core',
# )

######################################################################
## Define Dependencies
######################################################################

# Get the NumCpp Dependencies with/without library
dep_core = declare_dependency(
  compile_args: cython_cpp_flags,
  dependencies: _dep_list_numcpp,
  include_directories: inc_dir_core,
  # link_with: [_st_lib_core],  # Link with the static/shared library
  # link_args: ['-shared'],     # shared library that can be used by other programs at runtime.
)
# Optionally add a message to confirm the installation
message(
  '\nCpp Core Header dependency defined successfully: ' +
  '@0@ '.format('dep_core') +
  'with NumCpp: @0@'.format(get_option('lnpy_use_numcpp').enabled())
)

######################################################################
## Next
######################################################################

######################################################################
## Optional format, tidy
######################################################################

# # Custom targets (optional)
# custom_target('format',
#   command: ['clang-format', '-i', '-style=file:.clang-format', '@INPUT@'],
#   output: 'formatted',
#   input: files('NumCpp.hpp'),  # Adjust this pattern to match the actual header files
# )
# custom_target('tidy',
#   command: ['clang-tidy', '-p', meson.build_dir(), '-extra-arg=-std=c++17', '@INPUT@'],
#   output: 'tidy',
#   input: files('NumCpp.hpp'),  # Adjust this pattern to match the actual source files
# )