# LightNumPy

**Lightweight and Fast Numerical Computing Library for Python**

A lightweight version of NumPy (or similar functionality).

Install `lightnumpy` using `pip`:

```bash
# Placeholder, Not Implemented
pip install lightnumpy
```

---

## Why LightNumPy?
- **Performance-Driven**: Optimized for both CPU and hardware accelerators (GPU/TPU).
- **Focused**: Includes essential features without unnecessary overhead.
- **Adaptable**: Modular structure for customized extensions.
- **Scalable**: Ideal for IoT, embedded systems, and resource-limited devices.

---

## LightNumPy Project Structure

```sh
lightnumpy/  
│  
├── lightnumpy/                       # Core library source code  
│   ├── __init__.py                   # Main package initializer  
│   ├── .clang-format                 # Code formatting rules for C/C++, code formatting rules like braces placement, and spacing.
│   ├── .clang-tidy                   # Code linting rules for C/C++, code analysis, warnings, and bug detection.
│   ├── _c_core/                      # Low-level C implementation sources  
│   │   ├── include/                  # C headers  
│   │   │   ├── array.h               # Array definitions  
│   │   │   └── math_ops.h            # Math operation headers  
│   │   └── src/                      # C source files  
│   │       ├── array.c               # Array operations  
│   │       └── math_ops.c            # Math implementations  
│   ├── _cpp_core/                    # Higher-level C++ implementation sources  
│   │   ├── include/                  # C++ headers  
│   │   │   ├── tensor.hpp            # Tensor operations  
│   │   │   └── utilities.hpp         # Helper utilities  
│   │   └── src/                      # C++ source files  
│   │       ├── tensor.cpp            # Tensor operation implementations  
│   │       └── utilities.cpp         # Utility function implementations  
│   ├── _gpu_core/                    # GPU operations  
│   │   ├── include/                  # GPU headers  
│   │   │   ├── gpu_ops.hpp           # GPU operation definitions  
│   │   │   └── cuda_helpers.hpp      # CUDA helper utilities  
│   │   └── src/                      # GPU source files  
│   │       ├── gpu_ops.cu            # CUDA implementations  
│   │       └── cuda_helpers.cu       # CUDA utility functions  
│   ├── _tpu_core/                    # TPU operations  
│   │   ├── include/                  # TPU headers  
│   │   │   ├── tpu_ops.hpp           # TPU operation definitions  
│   │   │   └── tpu_helpers.hpp       # TPU helper utilities  
│   │   └── src/                      # TPU source files  
│   │       ├── tpu_ops.cpp           # TPU operation implementations (via XLA)  
│   │       └── tpu_helpers.cpp       # TPU utility functions  
│   ├── cy_bindings/                  # Cython implementation, Cython bridging native libraries with Python APIs
│   │   ├── __init__.py               # Initialize the cython package  
│   │   ├── include/                  # Headers  
│   │   └── src/                      # TPU source files  
│   │       ├── array_cy.pyx          # Cython implementation of array module  
│   │       ├── linalg_cy.pyx         # Cython implementation of linalg operations  
│   │       ├── utils_cy.pyx          # Cython utilities  
│   │       ├── gpu_cy.pyx            # GPU-specific Cython bindings  
│   │       ├── tpu_cy.pyx            # TPU-specific Cython bindings  
│   │       └── cython_helpers.pxd    # Shared Cython declarations (optional) 
│   ├── py_bindings/                  # Bindings for Python and native code, pybind11 bridging native libraries with Python APIs
│   │   ├── include/                  # Headers  
│   │   └── src/                      # TPU source files  
│   │       ├── c_bindings.c          # C-Python bindings  
│   │       ├── cpp_bindings.cpp      # C++-Python bindings  
│   │       ├── gpu_bindings.cu       # CUDA-Python bindings  
│   │       ├── tpu_bindings.cpp      # TPU-Python bindings  
│   │       └── pybind_utils.cpp      # Helper functions for bindings   
│   ├── python_api/                   # Pure Python layer providing user-friendly interfaces for core functionality
│   │   ├── __init__.py               # API entry point for `python_api`  
│   │   ├── _utils_impl.py            # get_c_include, get_cpp_include, and get_include for lightnumpy library's C and C++ headers
│   │   ├── array.py                  # Array class implementation and basic methods
│   │   ├── core.py                   # Contains core array functionality
│   │   ├── linalg.py                 # Basic linear algebra operations (e.g., dot, transpose)
│   │   ├── operations.py             # Element-wise operations (e.g., addition, multiplication) (CPU, GPU, TPU)  
│   │   ├── gpu_operations.py         # GPU-specific Python operations  
│   │   ├── tpu_operations.py         # TPU-specific Python operations  
│   │   └── utils.py                  # Utility functions for array manipulation
│   └── tests/                        # Core library tests  
│       ├── test_array.py             # Test for array module  
│       ├── test_tensor.py            # Test for tensor module  
│       ├── test_gpu_ops.py           # Test for GPU operations  
│       ├── test_tpu_ops.py           # Test for TPU operations
│       ├── test_cython_array.py      # Test for Cython array implementation  
│       └── test_cython_utils.py      # Test for Cython utility functions  
│       
├── examples/                         # Example usage and demos  
│   ├── array_example.py              # Example for arrays  
│   ├── tensor_example.py             # Example for tensors  
│   ├── gpu_example.py                # Example for GPU operations  
│   └── tpu_example.py                # Example for TPU operations  
│       
├── docs/                             # Documentation  
│   ├── index.md                      # Documentation index  
│   ├── api/                          # API reference  
│   │   ├── gpu_api.md                # GPU API documentation  
│   │   └── tpu_api.md                # TPU API documentation  
│   └── developer_guide.md            # Developer setup and guide  
│       
├── .github/                          # CI/CD configuration  
│   ├── issue_templates/              # GitHub issue templates  
│   └── workflows/                    # GitHub Actions workflows  
│       └── ci.yml                    # Main CI pipeline configuration  
│       
├── meson.build                       # Meson build configuration  
├── LICENSE                           # Project license
├── pyproject.toml                    # Python project configuration  
└── README.md                         # Project overview  
├── setup.cfg                         # Optional Python packaging configuration  
```


---

```sh
mkdir lightnumpy
cd lightnumpy

mkdir -p lightnumpy/{python_api,c_core/include,c_core/src,cpp_core/include,cpp_core/src,gpu_core/include,gpu_core/src,tpu_core/include,tpu_core/src,bindings,tests}

touch meson.build pyproject.toml README.md

mkdir -p .github/workflows
touch .github/workflows/ci.yml

mkdir docs
touch docs/index.md

mkdir examples
touch examples/{array_example.py,tensor_example.py,gpu_example.py,tpu_example.py}
```
---

## FAQs

#### Q: How does lightnumpy differ from numpy?
A: While numpy is a full-featured numerical library, lightnumpy is a lightweight version, designed to be faster and more efficient for specific use cases. It focuses on essential features, leaving out less commonly used ones.

#### Q: Is lightnumpy compatible with numpy?

A: For most common operations, yes! It provides a familiar API to help you transition smoothly.

#### Q: Can I use lightnumpy for GPU/TPU tasks?

A: Absolutely! lightnumpy has built-in support for hardware accelerators.

