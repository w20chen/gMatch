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
#include "cuda_runtime.h"

#define ull unsigned long long

static const int Zero = 0;
static const ull Zero_ull = 0;
static const int One = 1;

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


struct pint {
    int first, second;

    pint() {
        first = second = 0;
    }

    pint(int a, int b) {
        first = a;
        second = b;
    }
};


static pint
make_pint(int a, int b) {
    pint p(a, b);
    return p;
}


static void
swap(uint32_t &a, uint32_t &b) {
    uint32_t t = a;
    a = b;
    b = t;
}


static void
swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}


static void
TODO() {
    printf("Implement me!\n");
    assert(0);
}

static void
ERROR() {
    printf("Error!\n");
    assert(0);
}


static double
calculateMean(int *data, int num) {
    double sum = 0.;
    for (int i = 0; i < num; i++) {
        sum += data[i];
    }
    return sum / num;
}


static double
calculateVariance(int *data, int num, double mean) {
    double sumOfSquaresOfDifferences = 0.;
    for (int i = 0; i < num; i++) {
        sumOfSquaresOfDifferences += std::pow(data[i] - mean, 2);
    }
    return sumOfSquaresOfDifferences / num;
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
    if (nums == nullptr) {
        return false;
    }
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        int t = nums[mid];
        if (t > value) {
            high = mid - 1;
        }
        else if (t < value) {
            low = mid + 1;
        }
        else {
            return true;
        }
    }
    return false;
}


static __device__ __host__ int
binary_search_index(int *nums, int n, int value) {
    if (nums == nullptr) {
        return -1;
    }
    int low = 0;
    int high = n - 1;
    while (low <= high) {
        int mid = low + ((high - low) >> 1);
        int t = nums[mid];
        if (t > value) {
            high = mid - 1;
        }
        else if (t < value) {
            low = mid + 1;
        }
        else {
            return mid;
        }
    }
    return -1;
}


// Perform a parallel binary search using an entire warp (32 threads)
static __device__ bool
binary_search_warp(int *nums, int n, int value, int *warp_flags,
                   int warp_id_in_blk) {
    // Check if the input array is null
    if (nums == nullptr) {
        return false;
    }

    // Calculate the segment size for each thread in the warp
    int seg = n / warpSize;
    // Get the lane ID within the warp (0-31)
    int lane_id = threadIdx.x % warpSize;

    // If the segment size is 0, fall back to regular binary search
    if (seg == 0) {
        return binary_search(nums, n, value);
    }

    // Calculate the search range for this thread
    int low = seg * lane_id;
    int high = low + seg - 1;
    // The last thread in the warp handles any remaining elements
    if (lane_id + 1 == warpSize) {
        high = n - 1;
    }

    // Initialize the warp flag to false
    warp_flags[warp_id_in_blk] = false;

    // Synchronize all threads in the warp
    __syncwarp();

    // Perform binary search within each thread's segment
    while (low <= high) {
        // If any thread has found the value, exit the loop
        if (warp_flags[warp_id_in_blk] == true) {
            break;
        }
        // Calculate the midpoint
        int mid = low + ((high - low) >> 1);
        if (nums[mid] > value) {
            high = mid - 1;
        }
        else if (nums[mid] < value) {
            low = mid + 1;
        }
        else {
            // Value found, set the warp flag to true
            warp_flags[warp_id_in_blk] = true;
            break;
        }
    }

    // Synchronize all threads in the warp again
    __syncwarp();

    // Return the result of the search (true if found, false otherwise)
    return warp_flags[warp_id_in_blk];
}


static void __global__
print_partial_results(int *head, int col, int row) {
    if (row > 20) {
        return;
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", *(head + col * i + j));
        }
        printf("\n");
    }
    printf("\n");
}

static void
check_gpu_props() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available.\n");
        assert(deviceCount != 0);
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("Total global mem: %.2f MBytes\n",
               (float)prop.totalGlobalMem / 1048576.0f);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    }
}

void __device__ static
bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

static void
check_gpu_memory() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory: total %ld, available %ld\n", total_byte, free_byte);
}


#define ceil_div(a, b) (((a) + (b) - 1) / (b))
#define min_(a, b) ((a) > (b) ? (b) : (a))

static __host__ __device__ void
print_array(const char *name, int *arr, int len) {
    if (len <= 0) {
        return;
    }
    int flag = false;
    if (len > 80) {
        len = 80;
        flag = true;
    }
    int i = 0;
    printf("%s: ", name);
    for (; i + 1 < len; i++) {
        printf("%d,", arr[i]);
    }
    printf("%d", arr[i]);
    if (flag) {
        printf("...");
    }
    printf("\n");
}

static __host__ __device__ int
closestPowerOfTwo(int number) {
    number--;
    number |= number >> 1;
    number |= number >> 2;
    number |= number >> 4;
    number |= number >> 8;
    number |= number >> 16;
    number++;
    return number;
}

int __host__ static
countOnes(unsigned n) {
    int count = 0;
    while (n != 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

// Recording Time

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