# example here is inspired by Nvidia's blog post:
#   https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/
# code:
#   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples

# # test that optional initialization of cuda works to disable thin archives
# add_languages('cuda')

# # https://github.com/mesonbuild/meson/issues/13240
# # nvcc = meson.get_compiler('cuda').find_library('doesnotexist', required : false)
# # assert(not nvcc.found())
# nvcc = meson.get_compiler('cuda')
# cuda = import('unstable-cuda')

# arch_flags = cuda.nvcc_arch_flags(nvcc.version(), 'Common')

# message('NVCC version:   ' + nvcc.version())
# message('NVCC flags:     ' + ' '.join(arch_flags))

# # test device linking with -dc (which is equivalent to `--relocatable-device-code true`)
# _st_lib_nvcc = static_library('_st_lib_nvcc', ['src/b.cu'],
#   cuda_args : ['-dc'] + arch_flags,
# )


# arch_flags     = cuda.nvcc_arch_flags(nvcc.version(), 'Auto', detected: ['6.0'])
# arch_readable  = cuda.nvcc_arch_readable(nvcc.version(), 'Auto', detected: ['6.0'])
# driver_version = cuda.min_driver_version(nvcc.version())

# message('NVCC version:   ' + nvcc.version())
# message('NVCC flags:     ' + ' '.join(arch_flags))
# message('NVCC readable:  ' + ' '.join(arch_readable))
# message('Driver version: >=' + driver_version)

######################################################################
## Find CUDA and CuBLAS Dependencies
## https://github.com/mesonbuild/meson/tree/master/test%20cases/cuda
## pkg-config --list-all | grep cuda
######################################################################

# if host_machine.system() == 'linux'
#   # Linux: Check for CUDA or ROCm
#   # dep_cuda = dependency('cuda', modules: ['cublas'])
#   # dep_cuda = dependencies: dependency('cuda', modules: ['cublas']), link_language: 'cpp'
#   # dep_cuda = dependency('cuda', version: ['>=10.1'], required: false, disabler: true)
#   # dep_cuda = dependency('cuda', required: false)  # Try to find CUDA
#   # dep_rocm = dependency('rocm-opencl', required: false)  # Try to find ROCm OpenCL
#   cuda_dep = dependency('cuda', required: false)
#   rocm_dep = dependency('rocm-opencl', required: false)
#   # Check for dependencies
#   if cuda_dep.found()
#     gpu_library = declare_dependency(
#       include_directories: include_directories('/usr/local/cuda/include'),
#       link_with: [cuda_dep]
#     )
#     message('Using CUDA on Linux')
#   elif rocm_dep.found()
#     gpu_library = declare_dependency(
#       include_directories: include_directories('/opt/rocm/include'),
#       link_with: [rocm_dep]
#     )
#     message('Using ROCm on Linux')
#   else
#     error('No GPU library found on Linux!')
#   endif

# elif host_machine.system() == 'darwin'
#   # macOS: Use Metal or CUDA (if on older macOS)
#   if meson.version().version_compare('10.14')
#     metal_dep = dependency('metal', required: true)
#     gpu_library = declare_dependency(
#       include_directories: include_directories('/System/Library/Frameworks/Metal.framework/Headers'),
#       link_args: ['-framework', 'Metal']  # Link Metal framework
#     )
#     message('Using Metal on macOS')
#   else
#     cuda_dep = dependency('cuda', required: true)
#     gpu_library = declare_dependency(
#       include_directories: include_directories('/usr/local/cuda/include'),
#       link_with: [cuda_dep]
#     )
#     message('Using CUDA on macOS')
#   endif

# elif host_machine.system() == 'windows'
#   # Windows: Use CUDA only
#   cuda_dep = dependency('cuda', required: true)
#   gpu_library = declare_dependency(
#     include_directories: include_directories('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include'),
#     link_with: [cuda_dep]
#   )
#   message('Using CUDA on Windows')
# endif


######################################################################
## Define dependencies and Compiler Flags for gpu numC++
######################################################################

# Define args and dependencies for NumCpp
_global_gpu_numcpp_args = []
_dep_list_gpu_numcpp = []

if get_option('lnpy_use_gpu_numcpp').enabled()
  # Initialize an empty list for compiler flags
  _global_gpu_numcpp_args += [
    # The -D flag defines a macro that is globally available during compilation.
    # The -U flag removes a macro definition during compilation, effectively ignoring it.
    '-DLNPY_USE_GPU_NUMCPP',  # Enable all GPU NumCpp features
  ]
else
  _global_gpu_numcpp_args += [
    '-U LNPY_USE_GPU_NUMCPP',  # Disable all NumCpp features
  ]
endif

cython_cpp_flags += _global_gpu_numcpp_args

######################################################################
## Include the Headers directories for gpu _core, NumCpp
######################################################################

# include directories for Specific Subfolders
_gpu_core_inc_dir = [
  'src',
  'include',
]

######################################################################
## Define include_directories for gpu _core, NumCpp
######################################################################

# Use the include directory in your build setup
# Specify Include directories where your headers are located
# include_directories(header) -> static_library(mix), library(mix), declare_dependency(mix)
inc_dir_gpu_core = include_directories(
  _gpu_core_inc_dir
)
# Optionally add a message to confirm the installation
message(
  '\nCpp Core Header compatibility files defined successfully: ' +
  '@0@ '.format('lightnumpy/_gpu_core/include') +
  'with NumCpp: @0@'.format(get_option('lnpy_use_gpu_numcpp').enabled())
)

######################################################################
## NumCpp is a header-only library, which means it does not have any precompiled binary
## (like .so, .dll, or .a). 
######################################################################

# # Static library with C++ source file implementing bindings
# _st_lib_gpu_core = static_library('_st_lib_gpu_core', [],
#   include_directories: inc_dir_gpu_core,
#   dependencies: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'inlineshidden',
#   link_with: [],
#   link_args: [],
#   # install: true,
#   # install_dir: 'lightnumpy/_gpu_core',
# )
# # Shared (dynamic) library with C++ source file implementing bindings
# _dyn_lib_gpu_core = library('_dyn_lib_gpu_core', [],
#   include_directories: inc_dir_gpu_core,
#   dependencies: [],
#   c_args: [],
#   cpp_args: [],
#   gnu_symbol_visibility: 'hidden',
#   gnu_symbol_visibility: 'inlineshidden',
#   # link_with: [_st_lib_gpu_core],  # Link with the static library
#   # link_args: ['-shared'],  # shared library that can be used by other programs at runtime.
#   install: true,
#   install_dir: 'lightnumpy/_gpu_core',
# )

######################################################################
## Define Dependencies
######################################################################

# Get the NumCpp Dependencies with/without library
dep_gpu_core = declare_dependency(
  compile_args: cython_cpp_flags,
  dependencies: _dep_list_gpu_numcpp,
  include_directories: inc_dir_gpu_core,
  # link_with: [_st_lib_gpu_core],  # Link with the static/shared library
  # link_args: ['-shared'],     # shared library that can be used by other programs at runtime.
)
# Optionally add a message to confirm the installation
message(
  '\nCpp Core Header dependency defined successfully: ' +
  '@0@ '.format('dep_gpu_core') +
  'with NumCpp: @0@'.format(get_option('lnpy_use_gpu_numcpp').enabled())
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