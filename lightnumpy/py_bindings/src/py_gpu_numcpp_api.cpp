#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include "gpu_array_ops.cuh"


PYBIND11_MODULE(py_gpu_numcpp_api, m) {
    m.doc() = "Python bindings for GPU array operations using numC++";

    m.def("init_gpu_config", &init_gpu_config, py::arg("device_id"),
          "Initialize GPU configuration for the specified device.");

    m.def("create_gpu_array", [](int rows, int cols, float default_value) {
        return reinterpret_cast<uintptr_t>(create_gpu_array(rows, cols, default_value));
    }, py::arg("rows"), py::arg("cols"), py::arg("default_value"),
       "Create a GPU array with default values.");

    m.def("create_gpu_array_1d", [](py::array_t<float> data) {
        py::buffer_info buf = data.request();
        return reinterpret_cast<uintptr_t>(create_gpu_array_1d(static_cast<float*>(buf.ptr), buf.size));
    }, py::arg("data"), "Create a GPU array from a 1D NumPy array.");

    m.def("create_gpu_array_2d", [](py::array_t<float> data, int rows, int cols) {
        py::buffer_info buf = data.request();
        return reinterpret_cast<uintptr_t>(create_gpu_array_2d(static_cast<float*>(buf.ptr), rows, cols));
    }, py::arg("data"), py::arg("rows"), py::arg("cols"),
       "Create a GPU array from a 2D NumPy array.");

    m.def("get_gpu_array_metadata", [](uintptr_t array_ptr) {
        void* array = reinterpret_cast<void*>(array_ptr);
        return py::make_tuple(get_gpu_array_rows(array),
                              get_gpu_array_cols(array),
                              get_gpu_array_size(array));
    }, py::arg("array_ptr"), "Get metadata (rows, cols, size) of a GPU array.");

    m.def("delete_gpu_array", [](uintptr_t array_ptr) {
        delete_gpu_array(reinterpret_cast<void*>(array_ptr));
    }, py::arg("array_ptr"), "Clean up GPU array.");
}
