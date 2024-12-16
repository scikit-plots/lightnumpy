#include "gpu_array_ops.cuh"

// Initialize GPU configuration
void init_gpu_config(int device_id) {
    np::getGPUConfig(device_id);
}

// Create a GPU array with default values
void* create_gpu_array(int rows, int cols, float default_value) {
    return new np::ArrayGPU<float>(rows, cols, default_value);
}

// Create a GPU array from a 1D array
void* create_gpu_array_1d(const float* data, int size) {
    std::vector<float> vec(data, data + size);
    return new np::ArrayGPU<float>(vec);
}

// Create a GPU array from a 2D array
void* create_gpu_array_2d(const float* data, int rows, int cols) {
    std::vector<std::vector<float>> vec(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            vec[i][j] = data[i * cols + j];
        }
    }
    return new np::ArrayGPU<float>(vec);
}

// Get metadata from a GPU array
int get_gpu_array_rows(void* array) {
    return static_cast<np::ArrayGPU<float>*>(array)->rows;
}

int get_gpu_array_cols(void* array) {
    return static_cast<np::ArrayGPU<float>*>(array)->cols;
}

int get_gpu_array_size(void* array) {
    return static_cast<np::ArrayGPU<float>*>(array)->size();
}

float* get_gpu_array_data(void* array) {
    return static_cast<np::ArrayGPU<float>*>(array)->mat;
}

// Clean up GPU array
void delete_gpu_array(void* array) {
    delete static_cast<np::ArrayGPU<float>*>(array);
}
