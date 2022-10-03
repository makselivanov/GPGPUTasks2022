#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters, unsigned int smoothing)
{
    // DONE если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int idx = get_group_id(0);
    const unsigned int x_size = get_global_size(0);
    const unsigned int idy = get_group_id(1);
    const unsigned int y_size = get_global_size(1);
    const unsigned int dx = get_local_id(0);
    const unsigned int dx_size = get_local_size(0);
    const unsigned int dy = get_local_id(1);
    const unsigned int dy_size = get_local_size(1);

    float x0 = fromX + (idx * dx_size + dx + 0.5f) * sizeX / (x_size * dx_size);
    float y0 = fromY + (idy * dy_size + dy + 0.5f) * sizeY / (y_size * dy_size);

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;
    if (smoothing != 0 && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    __local float local_array[256];

    local_array[dx * dy_size + dy] = result;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (dx == 0 && dy == 0) {
        result = 0;
        for (int i = 0; i < dx_size * dy_size; ++i) {
            result += local_array[i];
        }
        result /= dx_size * dy_size;

        result = 1.0f * result / iters;
        results[idy * x_size + idx] = result;
    }
}
