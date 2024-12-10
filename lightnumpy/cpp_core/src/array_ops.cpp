#include <vector>

std::vector<double> add_arrays(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}
