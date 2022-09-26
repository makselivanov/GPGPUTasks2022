#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        //OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
        //what():  OpenCL error code -30 encountered

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize * sizeof(unsigned char), platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t vendorNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
        std::vector<unsigned char> vendorName(vendorNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize * sizeof(unsigned char), vendorName.data(), nullptr));
        std::cout << "    Vendor name: " << vendorName.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Number of all devices: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            cl_device_id& device_id = devices[deviceIndex];
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize);
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, deviceNameSize * sizeof(unsigned char), deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
            std::cout << "        Device type: ";
            bool flag = false;
            if (deviceType & CL_DEVICE_TYPE_CPU) {
                if (!flag) {
                    flag = true;
                }
                std::cout << "CPU";
            }
            if (deviceType & CL_DEVICE_TYPE_GPU) {
                if (!flag) {
                    flag = true;
                } else {
                    std::cout << ", ";
                };
                std::cout << "GPU";
            }
            if (!flag) {
                std::cout << "Unknown";
            }
            std::cout << std::endl;
            //CL_DEVICE_GLOBAL_MEM_SIZE
            cl_ulong deviceMemSize;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMemSize), &deviceMemSize,
                                          nullptr));
            std::cout << "        Global MEM size: " << deviceMemSize << std::endl;
            //CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
            cl_ulong deviceMemCache;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceMemCache), &deviceMemCache,
                                          nullptr));
            std::cout << "        Global MEM cache size: " << deviceMemCache << std::endl;
            //CL_DEVICE_GLOBAL_MEM_CACHE_TYPE
            cl_device_mem_cache_type deviceMemCacheType;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(deviceMemCacheType), &deviceMemCacheType,
                                          nullptr));
            std::cout << "        Global MEM cache type: ";
            switch (deviceMemCacheType) {
                case CL_READ_ONLY_CACHE:
                    std::cout << "Read only";
                    break;
                case CL_READ_WRITE_CACHE:
                    std::cout << "Read/write";
                    break;
                case CL_NONE:
                    std::cout << "None";
                    break;
            }
            std::cout << std::endl;
            //CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
            cl_uint deviceMemCacheLineSize;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(deviceMemCacheLineSize), &deviceMemCacheLineSize,
                                          nullptr));
            std::cout << "        Global MEM cache line size: " << deviceMemCacheLineSize << std::endl;
            //CL_DEVICE_LOCAL_MEM_SIZE
            cl_ulong deviceLocalMemSize;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(deviceLocalMemSize), &deviceLocalMemSize,
                                          nullptr));
            std::cout << "        Local MEM size: " << deviceLocalMemSize << std::endl;
            //CL_DEVICE_LOCAL_MEM_TYPE
            cl_device_local_mem_type deviceLocalMemType;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(deviceLocalMemType), &deviceLocalMemType,
                                          nullptr));
            std::cout << "        Local MEM type: ";
            switch (deviceLocalMemType) {
                case CL_LOCAL:
                    std::cout << "Local";
                    break;
                case CL_GLOBAL:
                    std::cout << "Global";
                    break;
                case CL_NONE:
                    std::cout << "None";
                    break;
            }
            std::cout << std::endl;
            //CL_DEVICE_MAX_MEM_ALLOC_SIZE
            cl_ulong deviceMaxMemAllocSize;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(deviceMaxMemAllocSize), &deviceMaxMemAllocSize,
                                          nullptr));
            std::cout << "        Max MEM alloc size: " << deviceMaxMemAllocSize << std::endl;
            //CL_DEVICE_ERROR_CORRECTION_SUPPORT
            cl_bool deviceErrorCorrectionSupport;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(deviceErrorCorrectionSupport), &deviceErrorCorrectionSupport,
                                          nullptr));
            std::cout << "        Error correction support: " << (deviceErrorCorrectionSupport == CL_TRUE ? "TRUE" : "FALSE") << std::endl;
        }
    }
    return 0;
}
