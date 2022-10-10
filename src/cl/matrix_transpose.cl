#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 32

__kernel void matrix_transpose(
        const __global float* matrix,
        __global float* results,
        const unsigned int m,
        const unsigned int k)
{
    const unsigned int indy = get_global_id(0); //from 0 to K with rounding
    const unsigned int indx = get_global_id(1); //from 0 to M with rounding
    const unsigned int local_y = get_local_id(0); //from 0 to work_group_size1
    const unsigned int local_x = get_local_id(1); //from 0 to work_group_size2
    const unsigned int local_group_y = get_group_id(0);
    const unsigned int local_group_x = get_group_id(1);
    const unsigned int local_size_y = get_local_size(0);
    const unsigned int local_size_x = get_local_size(1);
    __local float submatrix[TILE_SIZE * TILE_SIZE]; //<-- Because of banks conflicts?

    if (indx < k && indy < m) {
        submatrix[local_x * TILE_SIZE + local_y] = matrix[indx * k + indy];
    }

    float tmp = submatrix[local_x * TILE_SIZE + local_y];
    barrier(CLK_LOCAL_MEM_FENCE);
    submatrix[local_y * TILE_SIZE + local_x] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int new_local_x, new_local_y;
    unsigned int new_global_x, new_global_y;
    if (local_size_x != local_size_y) {
        const unsigned int index_in_group = local_x * local_size_y + local_y;
        new_local_x = (index_in_group / local_size_x);
        new_local_y = (index_in_group % local_size_x);
    } else {
        new_local_x = local_x;
        new_local_y = local_y;
    }
    new_global_x = local_group_y * local_size_y + new_local_x;
    new_global_y = local_group_x * local_size_x + new_local_y;
    if (new_global_x < m && new_global_y < k) {
        results[new_global_x * m + new_global_y] = submatrix[new_local_x * TILE_SIZE + new_local_y];
    }
}