#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include "math.h"
//#include "printf.h"
#endif

#line 6

/*unsigned int min(unsigned int a, unsigned int b) {
    return (a < b) ? a : b;
}

unsigned int max(unsigned int a, unsigned int b) {
    return (a > b) ? a : b;
}*/

__kernel void merge(
        __global const float *a,
        __global float *result,
        int n,
        int k) {
    const int idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    const int window = k * 2;
    const int left = idx / window * window;
    const int right = left + k;
    const int index = idx % window;

    const int size_left = min(k, n - left);
    const int size_right = min(k, n - right);
    if (size_right == 0) {
        result[idx] = a[idx];
        return;
    }
    int L = max(-1, index - size_right - 1), R = min(index, size_left);
    while (R - L > 1) {
        int M = (R + L) / 2;
        int left_index = M;
        int right_index = index - M - 1;
        if (a[left + left_index] < a[right + right_index]) {
            L = M;
        } else {
            R = M;
        }
    }
    if (R == index - size_right || (R != size_left && a[left + R] < a[right + index - R])) {
        result[idx] = a[left + R];
    } else {
        result[idx] = a[right + index - R];
    }
}