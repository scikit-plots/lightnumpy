project(
  'lightnumpy',
  'c', 'cpp', #'python',
  version: '0.1.0',
  meson_version: '>= 1.5.0',
  license: 'BSD-3-Clause',
  # license_files: [],
  default_options: [
   'buildtype=debugoptimized',
   'b_ndebug=if-release',
   'c_std=c11',
   'cpp_std=c++17',    
  ],
)
release_version = meson.project_version()

######################################################################
## C and C++ Compiler Setup
######################################################################

cc  = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

######################################################################
## Compiler Flags
######################################################################

# Initialize an empty list for compiler flags
_c_flags = []
_cpp_flags = []

######################################################################
## Public Headers
######################################################################

# Directories for headers
# include_directories = include_directories('c_operations', 'cpp_operations')

######################################################################
## Shared library
######################################################################

# C and C++ sources
# c_sources = files('c_operations/add.c', 'c_operations/subtract.c')
# cpp_sources = files('cpp_operations/add.cpp', 'cpp_operations/subtract.cpp')

# Create shared library from C and C++ code
# c_lib = shared_library('c_operations', c_sources)
# cpp_lib = shared_library('cpp_operations', cpp_sources)

######################################################################
## Python
######################################################################

# Import the Python module from Meson to interact with the Python installation
# https://mesonbuild.com/Python-module.html
mod_py = import('python')
# NOTE: with Meson >=0.64.0 we can add `pure: false` here and remove that line
# everywhere else, see https://github.com/mesonbuild/meson/pull/10783.
py = mod_py.find_installation(pure: false)

# Get the Python Dependencies
dep_py = py.dependency()

######################################################################
## python headers 'Python.h'
######################################################################

if not cc.has_header('Python.h', dependencies: dep_py)
  error('Cannot compile `Python.h`. ' +
    'Perhaps you need to install python-dev|python-devel')
endif

######################################################################
## Fs-module
## https://mesonbuild.com/Fs-module.html
######################################################################

# Copy the main __init__.py|pxd files to the build dir (needed for Cython)
fs = import('fs')

######################################################################
## Copy all the .py files
## https://mesonbuild.com/Reference-manual_functions.html#install_subdir
## https://mesonbuild.com/Installing.html#installing
######################################################################

# Install the main `lightnumpy` package without any `tests` folders
# Copy all the .py files to the install dir, rather than using
# py.install_sources and needing to list them explicitely one by one
install_subdir(
  'lightnumpy',
  install_dir: py.get_install_dir(),  # Folder relative to site-packages to install to
  install_tag: 'python-runtime',      # Tag to help categorize the install
  exclude_directories: [              # Directories to exclude from installation
    'tests',                          # Excludes the 'tests' directory
    'NumCpp',                         # Excludes the 'NumCpp' directory Submodule
  ],
)
# Optionally add a message to confirm the installation
message(
  '\nlightnumpy files installed successfully: ' +
  '@0@'.format(py.get_install_dir())
)

######################################################################
## Next
######################################################################

# Subdirectories for different cores
# subdir('lightnumpy/c_core')
subdir('lightnumpy/cpp_core')
# subdir('lightnumpy/gpu_core')
# subdir('lightnumpy/tpu_core')
# subdir('lightnumpy/bindings')