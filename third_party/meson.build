######################################################################
## Submodule NumCpp
######################################################################

if not fs.exists('NumCpp/README.md')
  error(
  'Missing the `NumCpp` submodule! ' + 
  'Run `git submodule update --init` to fix this.')
endif

######################################################################
## Copy Include Headers directories for to _cpp_core/include/NumCpp
######################################################################

# _numcpp_inc_dir = [
#   'NumCpp/include/NumCpp/Core/Internal',  # version
#   'NumCpp/include',
#   # 'NumCpp/include/NumCpp',
#   # 'NumCpp/include/NumCpp/Coordinates',
#   # 'NumCpp/include/NumCpp/Coordinates/ReferenceFrames',
#   # 'NumCpp/include/NumCpp/Coordinates/Transforms',
#   # 'NumCpp/include/NumCpp/Core',
#   # 'NumCpp/include/NumCpp/Core/Internal',
#   # 'NumCpp/include/NumCpp/DateTime',
#   # 'NumCpp/include/NumCpp/Filter',
#   # 'NumCpp/include/NumCpp/Filter/Boundaries',
#   # 'NumCpp/include/NumCpp/Filter/Boundaries/Boundaries1d',
#   # 'NumCpp/include/NumCpp/Filter/Boundaries/Boundaries2d',
#   # 'NumCpp/include/NumCpp/Filter/Filters',
#   # 'NumCpp/include/NumCpp/Filter/Filters/Filters1d',
#   # 'NumCpp/include/NumCpp/Filter/Filters/Filters2d',
#   # 'NumCpp/include/NumCpp/Functions',
#   # 'NumCpp/include/NumCpp/ImageProcessing',
#   # 'NumCpp/include/NumCpp/Integrate',
#   # 'NumCpp/include/NumCpp/Linalg',
#   # 'NumCpp/include/NumCpp/Linalg/svd',
#   # 'NumCpp/include/NumCpp/Logging',
#   # 'NumCpp/include/NumCpp/NdArray',
#   # 'NumCpp/include/NumCpp/Polynomial',
#   # 'NumCpp/include/NumCpp/PythonInterface',
#   # 'NumCpp/include/NumCpp/Random',
#   # 'NumCpp/include/NumCpp/Roots',
#   # 'NumCpp/include/NumCpp/Rotations',
#   # 'NumCpp/include/NumCpp/Special',
#   # 'NumCpp/include/NumCpp/Utils',
#   # 'NumCpp/include/NumCpp/Vector',
# ]

# `NumCpp` install to simplify import path for public Headers;
# should be updated whenever new files are added to `NumCpp`
# Copy all the .hpp files to the install dir, rather than using
# py.install_sources and needing to list them explicitely one by one
# ls '/opt/conda/lib/python3.11/site-packages/lightnumpy/_cpp_core/include/NumCpp/include'
install_subdir(
  'NumCpp/include',
  install_dir: py.get_install_dir()/'lightnumpy/_core/include/NumCpp',
  install_tag: 'python-runtime',
)
# Optionally add a message to confirm the installation
message(
  '\nNumCpp Header compatibility files installed successfully: ' +
  '@0@'.format(py.get_install_dir()/'lightnumpy/_core/include/NumCpp')
)

######################################################################
## Submodule boost-math
######################################################################

# if not fs.exists('math/README.md')
#   error('Missing the `math` submodule! Run `git submodule update --init` to fix this.')
# endif
# # Optionally add a message to confirm the installation
# message(
#   '\nmath compatibility files installed successfully: ' +
#   '@0@'.format('third_party/xla')
# )

######################################################################
## Submodule boost
######################################################################

# if not fs.exists('boost/README.md')
#   error('Missing the `boost` submodule! Run `git submodule update --init` to fix this.')
# endif
# # Optionally add a message to confirm the installation
# message(
#   '\nboost compatibility files installed successfully: ' +
#   '@0@'.format('third_party/xla')
# )

######################################################################
## Submodule xla
######################################################################

# if not fs.exists('xla/README.md')
#   error('Missing the `xla` submodule! Run `git submodule update --init` to fix this.')
# endif
# # Optionally add a message to confirm the installation
# message(
#   '\nXLA compatibility files installed successfully: ' +
#   '@0@'.format('third_party/xla')
# )

######################################################################
## 
######################################################################