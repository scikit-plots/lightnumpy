#pragma once  // Ensure the file is included only once by the compiler

#ifndef GPU_ARRAY_OPS_CUH
#define GPU_ARRAY_OPS_CUH

#include "numC++/npGPUArray.cuh"
#include "numC++/gpuConfig.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper for initializing GPU configuration
void init_gpu_config(int device_id);

// Create a GPU array with default values
void* create_gpu_array(int rows, int cols, float default_value);

// Create a GPU array from a 1D array
void* create_gpu_array_1d(const float* data, int size);

// Create a GPU array from a 2D array
void* create_gpu_array_2d(const float* data, int rows, int cols);

// Get GPU array metadata
int get_gpu_array_rows(void* array);
int get_gpu_array_cols(void* array);
int get_gpu_array_size(void* array);
float* get_gpu_array_data(void* array);

// Clean up GPU array
void delete_gpu_array(void* array);

#ifdef __cplusplus
}
#endif

#endif // GPU_ARRAY_OPS_CUH