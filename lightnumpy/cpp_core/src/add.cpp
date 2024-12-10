#include <iostream>
#include <vector>

/* Adds two arrays element-wise using C++ vectors */
void add_arrays(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
    size_t n = a.size();
    result.resize(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
