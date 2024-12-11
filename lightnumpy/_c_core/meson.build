######################################################################
## Include the Headers directories for _c_core
######################################################################

# list of include directories for Specific Subfolders
_c_core_inc_dir = [
  'src',
  'include',
]

######################################################################
## Define include_directories for _c_core
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_c_core = include_directories(
  _c_core_inc_dir
)
# Optionally add a message to confirm the installation
message(
  '\nC core Header compatibility files defined successfully: ' +
  '@0@'.format('lightnumpy/_c_core/include')
)

######################################################################
## Define library for _c_core
######################################################################

# # Static library with C source file implementing bindings
# _st_lib_c_core = static_library('_st_lib_c_core', [],
#   include_directories: inc_dir_c_core,
#   dependencies: [],
#   c_args: cython_c_flags,
#   cpp_args: cython_cpp_flags,
#   gnu_symbol_visibility: 'inlineshidden',
#   link_with: [],
#   link_args: [],
#   # install: true,
#   # install_dir: 'lightnumpy/_c_core',
# )
# # Shared (dynamic) library with C source file implementing bindings
# _dyn_lib_c_core = library('_dyn_lib_c_core', [],
#   include_directories: inc_dir_c_core,
#   dependencies: [],
#   c_args: cython_c_flags,
#   cpp_args: cython_cpp_flags,
#   gnu_symbol_visibility: 'hidden',
#   gnu_symbol_visibility: 'inlineshidden',
#   # link_with: [_st_lib_c_core],  # Link with the static library
#   # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
#   install: true,
#   install_dir: 'lightnumpy/_c_core',
# )

######################################################################
## Define Dependencies
######################################################################

# Get the Sources-Headers Dependencies
dep_c_core = declare_dependency(
  compile_args: cython_c_flags,
  dependencies: [],
  include_directories: inc_dir_c_core,
  # link_with: [_st_lib_c_core],  # Link with the static/shared library
  # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
)
# Optionally add a message to confirm the installation
message(
  '\nC core Header dependency defined successfully: ' +
  '@0@'.format('dep_c_core')
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