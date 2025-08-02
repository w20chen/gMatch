#include <iostream>
#include <vector>
#include <cstdint>

#define BUCKET_SIZE 4

static __device__ __host__ uint32_t
cuckoo_hash_func1(const uint32_t &k) {
    return k;
}

// Robert Jenkins' 32-bit integer hash function
// A well-known non-cryptographic hash function with excellent distribution properties
// Source: https://gist.github.com/badboy/6267743
static __device__ __host__ uint32_t
cuckoo_hash_func2(const uint32_t &k) {
    uint32_t a = k;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}


template<typename ValueType>
class CuckooHashGPU {
public:
    uint32_t *d_keys1;
    ValueType *d_values1;
    uint32_t *d_keys2;
    ValueType *d_values2;
    uint32_t size_minus_1;

public:
    __device__ ValueType *
    find(uint32_t key) {
#if (BUCKET_SIZE == 4)
        int pos1 = cuckoo_hash_func1(key) & size_minus_1;
        int4 data1 = reinterpret_cast<int4 *>(d_keys1)[pos1];
        int *bucket1 = (int *)&data1;

        for (int i = 0; i < 4; i++) {
            if (bucket1[i] == key) {
                return &d_values1[pos1 * BUCKET_SIZE + i];
            }
        }

        int pos2 = cuckoo_hash_func2(key) & size_minus_1;
        int4 data2 = reinterpret_cast<int4 *>(d_keys2)[pos2];
        int *bucket2 = (int *)&data2;

        for (int i = 0; i < 4; i++) {
            if (bucket2[i] == key) {
                return &d_values2[pos2 * BUCKET_SIZE + i];
            }
        }
#elif (BUCKET_SIZE == 2)
        int pos1 = cuckoo_hash_func1(key) & size_minus_1;
        int2 data1 = reinterpret_cast<int2 *>(d_keys1)[pos1];
        int *bucket1 = (int *)&data1;

        for (int i = 0; i < 2; i++) {
            if (bucket1[i] == key) {
                return &d_values1[pos1 * BUCKET_SIZE + i];
            }
        }

        int pos2 = cuckoo_hash_func2(key) & size_minus_1;
        int2 data2 = reinterpret_cast<int2 *>(d_keys2)[pos2];
        int *bucket2 = (int *)&data2;

        for (int i = 0; i < 2; i++) {
            if (bucket2[i] == key) {
                return &d_values2[pos2 * BUCKET_SIZE + i];
            }
        }
#elif (BUCKET_SIZE == 1)
        int pos1 = cuckoo_hash_func1(key) & size_minus_1;
        int data1 = d_keys1[pos1];
        if (data1 == key) {
            return &d_values1[pos1];
        }

        int pos2 = cuckoo_hash_func2(key) & size_minus_1;
        int data2 = d_keys2[pos2];
        if (data2 == key) {
            return &d_values2[pos2];
        }
#endif
        return nullptr;
    }
};


template<typename ValueType>
struct bucket {
    int size_;
    uint32_t keys[BUCKET_SIZE];
    ValueType values[BUCKET_SIZE];

    bucket() {
        for (int i = 0; i < BUCKET_SIZE; i++) {
            keys[i] = UINT32_MAX;
        }
        size_ = 0;
    }

    bool
    has_empty_cell() {
        return size_ < BUCKET_SIZE;
    }

    bool
    is_empty() {
        return size_ == 0;
    }

    void
    insert(uint32_t key, ValueType val) {
        keys[size_] = key;
        values[size_] = val;
        size_++;
    }

    int
    get_key_pos(uint32_t key) {
        for (int i = 0; i < size_; i++) {
            if (keys[i] == key) {
                return i;
            }
        }
        return -1;
    }

    std::pair<uint32_t *, ValueType *>
    random_choose() {
        int local_index = rand() % BUCKET_SIZE;
        return std::make_pair(keys + local_index, values + local_index);
    }
};


template<typename ValueType>
class CuckooHash {
    std::vector<bucket<ValueType>> hash_table1_;
    std::vector<bucket<ValueType>> hash_table2_;

    uint32_t size_;
    // Number of buckets in the hash table

public:
    explicit
    CuckooHash(uint32_t originSz = (128 / BUCKET_SIZE)) {
        size_ = roundUpPowerOf2(originSz | 2);
        // Ensure the table size is at least 2

        hash_table1_.resize(size_);
        hash_table2_.resize(size_);
    }

    ValueType
    find(const uint32_t &k) {
        uint32_t p1 = get_pos1(k);
        if (!hash_table1_[p1].is_empty()) {
            int idx = hash_table1_[p1].get_key_pos(k);
            if (idx >= 0) {
                return hash_table1_[p1].values[idx];
            }
        }

        uint32_t p2 = get_pos2(k);
        if (!hash_table2_[p2].is_empty()) {
            int idx = hash_table2_[p2].get_key_pos(k);
            if (idx >= 0) {
                return hash_table2_[p2].values[idx];
            }
        }

        return ValueType();
    }

    int
    size() {
        return 2 * size_ * BUCKET_SIZE;
    }

    void
    insert(uint32_t key, ValueType value) {

retry_after_rehash:

        uint32_t first_try_pos = get_pos1(key);
        bool to_table1 = true;

        if (hash_table1_[first_try_pos].has_empty_cell()) {
            hash_table1_[first_try_pos].insert(key, value);
            return;
        }

        auto victim = hash_table1_[first_try_pos].random_choose();
        std::swap(*victim.first, key);
        std::swap(*victim.second, value);
        to_table1 = false;

        std::pair<uint32_t, ValueType> cur_pair = std::make_pair(key, value);

        while (true) {
            if (to_table1) {
                uint32_t cur_pos = get_pos1(cur_pair.first);
                // Detect and break potential cycles: if we are trying to insert the same key
                // back into the same position in table1, we have formed a cycle
                if (cur_pos == first_try_pos && cur_pair.first == key && !hash_table1_[cur_pos].has_empty_cell()) {
                    break;
                }

                if (hash_table1_[cur_pos].has_empty_cell()) {
                    hash_table1_[cur_pos].insert(cur_pair.first, cur_pair.second);
                    return;
                }

                auto victim = hash_table1_[cur_pos].random_choose();
                std::swap(*victim.first, cur_pair.first);
                std::swap(*victim.second, cur_pair.second);
                to_table1 = false;
            }
            else {
                uint32_t cur_pos = get_pos2(cur_pair.first);
                if (hash_table2_[cur_pos].has_empty_cell()) {
                    hash_table2_[cur_pos].insert(cur_pair.first, cur_pair.second);
                    return;
                }

                auto victim = hash_table2_[cur_pos].random_choose();
                std::swap(*victim.first, cur_pair.first);
                std::swap(*victim.second, cur_pair.second);
                to_table1 = true;
            }
        }

        rehash();

        goto retry_after_rehash;
    }

private:

    inline uint32_t
    get_pos1(const uint32_t &k) {
        return cuckoo_hash_func1(k) & (size_ - 1);
    }

    inline uint32_t
    get_pos2(const uint32_t &k) {
        return cuckoo_hash_func2(k) & (size_ - 1);
    }

    unsigned int
    roundUpPowerOf2(uint32_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    // Creates a new cuckoo hash table with twice the size and reinserts all elements
    // Multiple recursive rehashes may be necessary before insertion succeeds
    void
    rehash() {
        CuckooHash<ValueType> new_hash(size_ * 2);

        // for each bucket in hash table 1 and 2
        for (int i = 0; i < size_; i++) {
            // for each cell in the bucket
            for (int j = 0; j < hash_table1_[i].size_; j++) {
                new_hash.insert(hash_table1_[i].keys[j], hash_table1_[i].values[j]);
            }
            for (int j = 0; j < hash_table2_[i].size_; j++) {
                new_hash.insert(hash_table2_[i].keys[j], hash_table2_[i].values[j]);
            }
        }

        hash_table1_ = new_hash.hash_table1_;
        hash_table2_ = new_hash.hash_table2_;
        size_ = new_hash.size_;
    }

public:
    int
    to_gpu(CuckooHashGPU<ValueType> *d_C) {
        CuckooHashGPU<ValueType> h_C;
        size_t total_size = size_ * BUCKET_SIZE;

        cudaMalloc(&h_C.d_keys1, total_size * sizeof(uint32_t));
        cudaMalloc(&h_C.d_values1, total_size * sizeof(ValueType));
        cudaMalloc(&h_C.d_keys2, total_size * sizeof(uint32_t));
        cudaMalloc(&h_C.d_values2, total_size * sizeof(ValueType));
        h_C.size_minus_1 = size_ - 1;

        std::vector<uint32_t> gpu_keys1(total_size, UINT32_MAX);
        std::vector<ValueType> gpu_values1(total_size);
        std::vector<uint32_t> gpu_keys2(total_size, UINT32_MAX);
        std::vector<ValueType> gpu_values2(total_size);

        for (int i = 0; i < size_; i++) {
            for (int j = 0; j < hash_table1_[i].size_; j++) {
                gpu_keys1[i * BUCKET_SIZE + j] = hash_table1_[i].keys[j];
                gpu_values1[i * BUCKET_SIZE + j] = hash_table1_[i].values[j];
            }
            for (int j = 0; j < hash_table2_[i].size_; j++) {
                gpu_keys2[i * BUCKET_SIZE + j] = hash_table2_[i].keys[j];
                gpu_values2[i * BUCKET_SIZE + j] = hash_table2_[i].values[j];
            }
        }

        cudaMemcpy(h_C.d_keys1, gpu_keys1.data(), total_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_C.d_values1, gpu_values1.data(), total_size * sizeof(ValueType), cudaMemcpyHostToDevice);
        cudaMemcpy(h_C.d_keys2, gpu_keys2.data(), total_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(h_C.d_values2, gpu_values2.data(), total_size * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemcpy(d_C, &h_C, sizeof(CuckooHashGPU<ValueType>), cudaMemcpyHostToDevice);

        int memory_size = 2 * total_size * (sizeof(uint32_t) + sizeof(ValueType));
        return memory_size;
    }
};