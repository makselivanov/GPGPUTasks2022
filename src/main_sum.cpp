#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    //DONE 6/6
    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        unsigned sum = 0;
        gpu::gpu_mem_32u as_gpu, result;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result.resizeN(1);
        const unsigned int workGroupSize = 256;
        {
            //making baseline
            const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            ocl::Kernel sum_baseline(sum_kernel, sum_kernel_length, "sum_baseline");
            sum_baseline.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum = 0;
                result.writeN(&sum, 1);
                sum_baseline.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                  as_gpu, n, result);
                result.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU baseline result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU baseline:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU baseline:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            //making bad loop
            const unsigned int valuePerWorkItem = 64;
            const unsigned int global_work_size = (((n + valuePerWorkItem - 1) / valuePerWorkItem) + workGroupSize - 1)
                                                  / workGroupSize * workGroupSize;
            ocl::Kernel sum_loop_bad(sum_kernel, sum_kernel_length, "sum_loop_bad");
            sum_loop_bad.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum = 0;
                result.writeN(&sum, 1);
                sum_loop_bad.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                  as_gpu, n, result);
                result.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU bad loop  result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU bad loop:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU bad loop:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            //making loop coalesced
            const unsigned int valuePerWorkItem = 64;
            const unsigned int global_work_size = (((n + valuePerWorkItem - 1) / valuePerWorkItem) + workGroupSize - 1)
                                                  / workGroupSize * workGroupSize;
            ocl::Kernel sum_loop_coalesced(sum_kernel, sum_kernel_length, "sum_loop_coalesced");
            sum_loop_coalesced.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum = 0;
                result.writeN(&sum, 1);
                sum_loop_coalesced.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                          as_gpu, n, result);
                result.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU coalesced loop result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU coalesced loop:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU coalesced loop:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            //making local memory
            const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            ocl::Kernel sum_local_memory(sum_kernel, sum_kernel_length, "sum_local_memory");
            sum_local_memory.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum = 0;
                result.writeN(&sum, 1);
                sum_local_memory.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                        as_gpu, n, result);
                result.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU local memory result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU local memory:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU local memory:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            //making tree
            const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            ocl::Kernel sum_tree(sum_kernel, sum_kernel_length, "sum_tree");
            sum_tree.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum = 0;
                result.writeN(&sum, 1);
                sum_tree.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                      as_gpu, n, result);
                result.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU tree result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU tree:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
