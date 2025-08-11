#ifndef HELPER_H
#define HELPER_H

#include <cstdio>
#include <string>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>
#include <assert.h>
#include <chrono>
#include <cmath>

#define ull unsigned long long

static const int Zero = 0;
static const ull Zero_ull = 0;


#define cudaCheck(call)                               \
do {                                                  \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess) {                  \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


static double
calculateMean(ull *data, int num) {
    double sum = 0.;
    int zero_cnt = 0;
    for (int i = 0; i < num; i++) {
        if (data[i] == 0) {
            zero_cnt++;
            continue;
        }
        sum += data[i];
    }
    return sum / (num - zero_cnt);
}

static double
calculateVariance(ull *data, int num, double mean) {
    double sumOfSquaresOfDifferences = 0.;
    int zero_cnt = 0;
    for (int i = 0; i < num; i++) {
        if (data[i] == 0) {
            zero_cnt++;
            continue;
        }
        sumOfSquaresOfDifferences += std::pow(data[i] - mean, 2);
    }
    return sumOfSquaresOfDifferences / (num - zero_cnt);
}

static ull
getMin(ull *data, int num) {
    ull min = data[0];
    for (int i = 1; i < num; i++) {
        if (data[i] == 0) {
            continue;
        }
        if (min > data[i]) {
            min = data[i];
        }
    }
    return min;
}

static ull
getMax(ull *data, int num) {
    ull max = data[0];
    for (int i = 1; i < num; i++) {
        if (data[i] == 0) {
            continue;
        }
        if (max < data[i]) {
            max = data[i];
        }
    }
    return max;
}

static void
printStatistics(const char *name, ull *arr, int len) {
    printf("========================================\n");
    if (name != nullptr) {
        printf("%s:\n", name);
    }

    double mean = calculateMean(arr, len);
    double variance = calculateVariance(arr, len, mean);
    printf("Mean: %.2lf\tVariance: %g\n", mean, variance);

    ull min_ = getMin(arr, len);
    ull max_ = getMax(arr, len);
    printf("Min: %lld\tMax: %lld\tMax-Min: %lld\n", min_, max_, max_ - min_);

    double coefficientOfVariation = sqrt(variance) / mean;
    printf("CV: %.4lf\n", coefficientOfVariation);
}

class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            tokens_.emplace_back(argv[i]);
        }
    }

    std::string
    get_cmd_option(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr =  std::find(tokens_.begin(), tokens_.end(), option);
        if (itr != tokens_.end() && ++itr != tokens_.end()) {
            return *itr;
        }
        return "";
    }

    bool
    check_cmd_option_exists(const std::string &option) const {
        return std::find(tokens_.begin(), tokens_.end(), option)
               != tokens_.end();
    }

    std::string
    get_cmd() {
        return std::accumulate(tokens_.begin(), tokens_.end(), std::string(" "));
    }

private:
    std::vector<std::string> tokens_;
};


static __device__ __host__ bool
binary_search(int *nums, int n, int value) {
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = ((low + high) >> 1);
        int t = nums[mid];
        bool cond = t >= value;
        high = cond ? (mid - 1) : high;
        low = cond ? low : (mid + 1);
    }
    return (low < n) && (nums[low] == value);
}


static void
check_gpu_props() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available.\n");
        assert(0);
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("========================================\n");
        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("Total global memory: %.2f MBytes\n",
               (float)prop.totalGlobalMem / 1048576.0f);
        printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("========================================\n");
    }
}

static int
selectDeviceWithMaxFreeMemory() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices available.\n");
        return -1;
    }

    size_t max_free = 0;
    int best_device = -1;

    for (int i = 0; i < device_count; i++) {
        err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        size_t free, total;
        err = cudaMemGetInfo(&free, &total);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get memory info for device %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        if (free > max_free) {
            max_free = free;
            best_device = i;
        }
    }

    if (best_device == -1) {
        fprintf(stderr, "No usable CUDA devices found.\n");
        return -1;
    }

    err = cudaSetDevice(best_device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set best device %d: %s\n", best_device, cudaGetErrorString(err));
        return -1;
    }

    return best_device;
}

static void
check_gpu_memory() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory: total %ld MB, available %ld MB\n", total_byte / 1024 / 1024, free_byte / 1024 / 1024);
}


#define ceil_div(a, b) (((a) + (b) - 1) / (b))


#define TIME_INIT() cudaEvent_t gpu_start, gpu_end;             \
    float kernel_time, total_kernel = 0.0, total_host = 0.0;    \
    auto cpu_start = std::chrono::high_resolution_clock::now(); \
    auto cpu_end = std::chrono::high_resolution_clock::now();   \
    std::chrono::duration<double> diff = cpu_end - cpu_start;

#define TIME_START()                                            \
    cpu_start = std::chrono::high_resolution_clock::now();      \
    cudaEventCreate(&gpu_start);                                \
    cudaEventCreate(&gpu_end);                                  \
    cudaEventRecord(gpu_start);

#define TIME_END()                                              \
    cpu_end = std::chrono::high_resolution_clock::now();        \
    cudaEventRecord(gpu_end);                                   \
    cudaEventSynchronize(gpu_start);                            \
    cudaEventSynchronize(gpu_end);                              \
    cudaEventElapsedTime(&kernel_time, gpu_start, gpu_end);     \
    total_kernel += kernel_time;                                \
    diff = cpu_end - cpu_start;                                 \
    total_host += diff.count();

#define PRINT_LOCAL_TIME(name)                                  \
    std::cout << "# " << name << ", time (ms): "                \
    << static_cast<unsigned long>(diff.count() * 1000)          \
    << "(host), "                                               \
    << static_cast<unsigned long>(kernel_time)                  \
    << "(kernel)\n";

#define PRINT_TOTAL_TIME(name)                                  \
    std::cout << "# " << name << " time (ms): "                 \
    << static_cast<unsigned long>(total_host * 1000)            \
    << "(host) "                                                \
    << static_cast<unsigned long>(total_kernel)                 \
    << "(kernel)\n";


#endif