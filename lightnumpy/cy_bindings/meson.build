######################################################################
## cython tree
######################################################################

# Copy the main "__init__.py"|"*.pxd" files to the build dir (needed for Cython)
# The Cython files that need to be processed, including dependencies and includes
cy_bindings_cython_tree = [
  # `_cython_tree` should have been defined earlier in the build script
  # to make sure scikitplot/__init__.py is copied
  fs.copyfile('__init__.py'),  # Ensure __init__.py is copied early
  fs.copyfile('cy_numcpp_api.pxd'),
  fs.copyfile('cy_numcpp_api.pxi'),
]

######################################################################
## Define dependencies and Compiler Flags
######################################################################

# Define dependencies with NumCpp
dep_list = [dep_py, dep_pybind11, dep_np, dep_core]

######################################################################
## extension metadata
#####################################################################
# Define metadata for shared Cython files '.pyx' extensions targeting both C and C++
cy_bindings_extension_metadata = {
  # Define Python Module Name
  'cy_numcpp_api':
    {
      'sources': [                                                 # .pyx source file with NumCpp + Cython
        # gen_cython_cpp.process('cy_numcpp_api.pyx'),             #  manually call a generator
        'cy_numcpp_api.pyx'
      ],
      'include_directories': [                                     # Include dirs for compilation
        inc_dir_core,    
      ],
      'dependencies': dep_list,                                    # External libraries and dependencies
      'link_with': [],                                             # Link with the created static library
      'override_options': [
        'cython_language=cpp',                                     # Ensure Cython knows to generate C++ code
        'optimization=3'                                           # Optimization level '-O3'
      ],
      'cython_args': cython_cpp_args,
      'c_args': cython_c_flags,                                    # Additional C/C++ arguments
      'cpp_args': cython_cpp_flags,                                # Additional C/C++ arguments
      'install': true,                                             # Whether to install the .so file executable after building
      'subdir': 'lightnumpy/cy_bindings',                          # Path where the module is located
    },
}

# https://mesonbuild.com/Syntax.html#foreach-with-a-dictionary
# Loop over each defined extension and create the corresponding module
foreach ext_name, ext_dict : cy_bindings_extension_metadata
  pyext_module = py.extension_module(
    ext_name,                                                      # The name of the extension module
    ext_dict.get('sources') + cy_bindings_cython_tree,             # Sources and dependencies
    include_directories: ext_dict.get('include_directories', []),  # Include directories
    dependencies: ext_dict.get('dependencies', []),                # Additional dependencies if any
    link_with: ext_dict.get('link_with', []),                      # Libraries to link with
    override_options : ext_dict.get('override_options', []),       # Options to override defaults
    cython_args: ext_dict.get('cython_args', []),                  # Use Cython specific arguments if any
    c_args: ext_dict.get('c_args', []),                            # Additional C compilation arguments
    cpp_args: ext_dict.get('cpp_args', []),                        # Additional C++ compilation arguments
    install: ext_dict.get('install', true),                        # Install the .so file this extension module
    subdir: ext_dict.get('subdir', '.'),                           # Subdirectory where the .so file module will be placed
    # install_dir: ext_dict.get('install_dir', '.'),               # Subdirectory where the .so file module will be installed
  )
endforeach