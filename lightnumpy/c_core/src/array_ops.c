#include <stddef.h>

void add_arrays(const double* a, const double* b, double* result, size_t size) {
    for (size_t i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}
