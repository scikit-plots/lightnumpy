#include <stdio.h>
#include <stdlib.h>

/* Adds two arrays element-wise */
void add_arrays(double* a, double* b, double* result, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
