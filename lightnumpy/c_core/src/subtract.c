#include <stdio.h>
#include <stdlib.h>

/* Subtracts two arrays element-wise */
void subtract_arrays(double* a, double* b, double* result, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}
