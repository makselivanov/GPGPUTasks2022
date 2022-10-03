#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_baseline(
        __global const unsigned int* a, unsigned int n,
        __global       unsigned int* r)
{
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;
    unsigned int res = atomic_add(r, a[index]);
}

#define VALUES_PER_WORK_ITEM 64u
__kernel void sum_loop_bad(
        __global const unsigned int* a, unsigned int n,
        __global       unsigned int* r)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);
    unsigned int res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        // V potential overflow if too big group_id * group_size
        const unsigned int index = group_id * group_size * VALUES_PER_WORK_ITEM + local_id * VALUES_PER_WORK_ITEM + i;
        if (index >= n) {
            break;
        }
        res += a[index];
    }
    if (res > 0) {
        atomic_add(r, res);
    }
}

__kernel void sum_loop_coalesced(
        __global const unsigned int* a, unsigned int n,
        __global       unsigned int* r)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);
    unsigned int res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        // V potential overflow if too big group_id * group_size
        const unsigned int index = group_id * group_size * VALUES_PER_WORK_ITEM + i * group_size + local_id;
        if (index >= n) {
            break;
        }
        res += a[index];
    }
    if (res > 0) {
        atomic_add(r, res);
    }
}

#define WORK_GROUP_SIZE 256u

__kernel void sum_local_memory(
        __global const unsigned int* a, unsigned int n,
        __global       unsigned int* r)
{
    __local unsigned int local_array[WORK_GROUP_SIZE];
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);
    if (global_id < n) {
        local_array[local_id] = a[global_id];
    } else {
        local_array[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unsigned int res = 0;
        for (unsigned int i = 0; i < get_local_size(0); ++i) {
            res += local_array[i];
        }
        atomic_add(r, res);
    }
}

__kernel void sum_tree(
        __global const unsigned int* a, unsigned int n,
        __global       unsigned int* r)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);

    __local unsigned int local_array[WORK_GROUP_SIZE];
    if (global_id < n) {
        local_array[local_id] = a[global_id];
    } else {
        local_array[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int nvalues = WORK_GROUP_SIZE; nvalues > 1u; nvalues >>= 1) {
        if (2 * local_id < nvalues) {
            unsigned int a = local_array[local_id];
            unsigned int b = local_array[local_id + nvalues / 2];
            local_array[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(r, local_array[0]);
    }
}

