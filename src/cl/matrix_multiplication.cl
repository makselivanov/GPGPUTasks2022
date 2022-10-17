#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16

inline float getOrDefault(const __global float *arr, int i, int j, int n, int m) {
    if (i < n && j < m)
        return arr[i * m + j];
    return 0;
}

inline void set(__global float *arr, int i, int j, int n, int m, int value) {
    if (i < n && j < m) {
        arr[i * m + j] = value;
    }
}

__kernel void matrix_multiplication(
        const __global float* a,
        const __global float* b,
        __global float* results,
        const unsigned int m,
        const unsigned int k,
        const unsigned int n)
{
    const unsigned int indn = get_global_id(0);   //from 0 to N with rounding
    const unsigned int indm = get_global_id(1);   //from 0 to M with rounding
    const unsigned int local_n = get_local_id(0); //from 0 to TILE_SIZE
    const unsigned int local_m = get_local_id(1); //from 0 to TILE_SIZE
    const unsigned int local_group_n = get_group_id(0);
    const unsigned int local_group_m = get_group_id(1);
    const unsigned int local_size_n = get_local_size(0);
    const unsigned int local_size_m = get_local_size(1);
    __local float suba[TILE_SIZE + 1][TILE_SIZE + 1];
    __local float subb[TILE_SIZE + 1][TILE_SIZE + 1];
    float result = 0.f;
    for (unsigned int index = 0; index < (k + TILE_SIZE - 1); index += TILE_SIZE) {
        const unsigned int shift = index;

        suba[local_m][local_n] = getOrDefault(a, indm, shift + local_n, m, k);
        subb[local_m][local_n] = getOrDefault(b, shift + local_m, indn, k, n);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int i = 0; i < TILE_SIZE; ++i) {
            result += suba[local_m][i] * subb[i][local_n];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    set(results, indm, indn, m, n, result);
}

#define TILE_SIZE2 32
#define THREAD_WORK 16

__kernel void matrix_multiplication_fma(
        const __global float* a,
        const __global float* b,
        __global float* results,
        const unsigned int m,
        const unsigned int k,
        const unsigned int n)
{
    const unsigned int indn = get_global_id(0);   //from 0 to N / THREAD_WORK with rounding
    const unsigned int indm = get_global_id(1);   //from 0 to M with rounding
    const unsigned int local_n = get_local_id(0); //from 0 to TILE_SIZE / THREAD_WORK
    const unsigned int local_m = get_local_id(1); //from 0 to TILE_SIZE
    const unsigned int local_group_n = get_group_id(0);
    const unsigned int local_group_m = get_group_id(1);
    const unsigned int local_size_n = get_local_size(0);
    const unsigned int local_size_m = get_local_size(1);
    __local float suba[TILE_SIZE2][TILE_SIZE2];
    __local float subb[TILE_SIZE2][TILE_SIZE2];
    float sum[THREAD_WORK];
    for (unsigned int index = 0; index < THREAD_WORK; ++index)
        sum[index] = 0;
    const unsigned int new_local_n = local_n * THREAD_WORK;
    for (unsigned int index = 0; index < (k + TILE_SIZE - 1); index += TILE_SIZE) {
        const unsigned int shift = index;

        for (unsigned int w = 0; w < THREAD_WORK; ++w)
            suba[new_local_n + w][local_m] = getOrDefault(a, indm, shift + new_local_n + w, m, k);
        for (unsigned int w = 0; w < THREAD_WORK; ++w)
            subb[new_local_n + w][local_m] = getOrDefault(b, shift + local_m, indn * THREAD_WORK + w, k, n);

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int i = 0; i < TILE_SIZE; ++i) {
            float tmp = suba[i][local_m];
            for (unsigned int w = 0; w < THREAD_WORK; ++w) {
                sum[w] += tmp * subb[new_local_n + w][i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (unsigned int w = 0; w < THREAD_WORK; ++w) {
        set(results, indm, indn * THREAD_WORK + w, m, n, sum[w]);
    }
}