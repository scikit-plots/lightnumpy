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

## Project Structure

```sh
lightnumpy/  
│  
├── lightnumpy/                  # Core library source code  
│   ├── __init__.py              # Main package initializer  
│   ├── python_api/              # Python API module  
│   │   ├── __init__.py          # API entry point for `python_api`  
│   │   ├── array.py             # Array class implementation and basic methods
│   │   ├── core.py              # Contains core array functionality
│   │   ├── linalg.py            # Basic linear algebra operations (e.g., dot, transpose)
│   │   ├── operations.py        # Element-wise operations (e.g., addition, multiplication) (CPU, GPU, TPU)  
│   │   ├── gpu_operations.py    # GPU-specific Python operations  
│   │   ├── tpu_operations.py    # TPU-specific Python operations  
│   │   └── utils.py             # Utility functions for array manipulation
│   ├── c_core/                  # C implementation sources  
│   │   ├── include/             # C headers  
│   │   │   ├── array.h          # Array definitions  
│   │   │   └── math_ops.h       # Math operation headers  
│   │   └── src/                 # C source files  
│   │       ├── array.c          # Array operations  
│   │       └── math_ops.c       # Math implementations  
│   ├── cpp_core/                # C++ implementation sources  
│   │   ├── include/             # C++ headers  
│   │   │   ├── tensor.hpp       # Tensor operations  
│   │   │   └── utilities.hpp    # Helper utilities  
│   │   └── src/                 # C++ source files  
│   │       ├── tensor.cpp       # Tensor operation implementations  
│   │       └── utilities.cpp    # Utility function implementations  
│   ├── gpu_core/                # GPU operations  
│   │   ├── include/             # GPU headers  
│   │   │   ├── gpu_ops.hpp      # GPU operation definitions  
│   │   │   └── cuda_helpers.hpp # CUDA helper utilities  
│   │   └── src/                 # GPU source files  
│   │       ├── gpu_ops.cu       # CUDA implementations  
│   │       └── cuda_helpers.cu  # CUDA utility functions  
│   ├── tpu_core/                # TPU operations  
│   │   ├── include/             # TPU headers  
│   │   │   ├── tpu_ops.hpp      # TPU operation definitions  
│   │   │   └── tpu_helpers.hpp  # TPU helper utilities  
│   │   └── src/                 # TPU source files  
│   │       ├── tpu_ops.cpp      # TPU operation implementations (via XLA)  
│   │       └── tpu_helpers.cpp  # TPU utility functions  
│   ├── bindings/                # Bindings for Python and native code  
│   │   ├── c_bindings.c         # C-Python bindings  
│   │   ├── cpp_bindings.cpp     # C++-Python bindings  
│   │   ├── gpu_bindings.cu      # CUDA-Python bindings  
│   │   ├── tpu_bindings.cpp     # TPU-Python bindings  
│   │   └── pybind_utils.cpp     # Helper functions for bindings  
│   └── tests/                   # Core library tests  
│       ├── test_array.py        # Test for array module  
│       ├── test_tensor.py       # Test for tensor module  
│       ├── test_gpu_ops.py      # Test for GPU operations  
│       └── test_tpu_ops.py      # Test for TPU operations  
│  
├── examples/                    # Example usage and demos  
│   ├── array_example.py         # Example for arrays  
│   ├── tensor_example.py        # Example for tensors  
│   ├── gpu_example.py           # Example for GPU operations  
│   └── tpu_example.py           # Example for TPU operations  
│  
├── .github/                     # CI/CD configuration  
│   ├── workflows/               # GitHub Actions workflows  
│   │   └── ci.yml               # Main CI pipeline configuration  
│   └── issue_templates/         # GitHub issue templates  
│  
├── docs/                        # Documentation  
│   ├── index.md                 # Documentation index  
│   ├── api/                     # API reference  
│   │   ├── gpu_api.md           # GPU API documentation  
│   │   └── tpu_api.md           # TPU API documentation  
│   └── developer_guide.md       # Developer setup and guide  
│  
├── meson.build                  # Meson build configuration  
├── pyproject.toml               # Python project configuration  
├── setup.cfg                    # Optional Python packaging configuration  
└── README.md                    # Project overview  
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

