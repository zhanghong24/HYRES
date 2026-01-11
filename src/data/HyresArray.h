#pragma once

#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <cstdlib>

// 引入我们的通用类型定义和宏
#include "common/Types.h"

// 如果开启了 CUDA，引入运行时 API
#ifdef HYRES_USE_CUDA
  #include <cuda_runtime.h>
#endif

namespace Hyres {

namespace detail {
    inline void throw_if(bool cond, const std::string& msg) {
        if (cond) throw std::runtime_error(msg);
    }

    // 安全乘法，防止溢出
    inline std::size_t mul_overflow_safe(std::size_t a, std::size_t b) {
        if (a == 0 || b == 0) return 0;
        if (a > (std::numeric_limits<std::size_t>::max)() / b) {
            throw std::overflow_error("HyresArray size overflow");
        }
        return a * b;
    }

    // 内存对齐分配 (64字节对齐，利用 AVX512 优势)
    inline void* aligned_malloc(std::size_t alignment, std::size_t bytes) {
        if (bytes == 0) return nullptr;
        void* p = nullptr;
#if defined(_MSC_VER)
        p = _aligned_malloc(bytes, alignment);
#else
        if (posix_memalign(&p, alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#endif
        return p;
    }

    inline void aligned_free(void* p) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

#ifdef HYRES_USE_CUDA
    inline void cuda_check(cudaError_t e, const char* file, int line) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) +
                                     " at " + file + ":" + std::to_string(line));
        }
    }
    #define HYRES_CUDA_CHECK(call) ::Hyres::detail::cuda_check((call), __FILE__, __LINE__)
#endif
}

/**
 * @brief HyresArray: 高性能 Host-Device 混合数组
 * * 特性:
 * 1. 自动管理 Host (CPU) 内存，64字节对齐。
 * 2. 懒加载 (Lazy Allocation) Device (GPU) 内存。
 * 3. 索引顺序: First dimension varies fastest (i-fastest)，兼容 Fortran。
 * 对于 3D 数组 (ni, nj, nk)，索引 (i,j,k) 的 offset = i + ni*j + ni*nj*k。
 */
template <typename T>
class HyresArray {
public:
    using value_type = T;

    HyresArray() = default;

    // 变长参数构造函数，例如: HyresArray<double> arr(ni, nj, nk);
    template <typename... Dims, typename = std::enable_if_t<(std::is_integral_v<Dims> && ...)>>
    explicit HyresArray(Dims... dims) {
        resize(static_cast<std::size_t>(dims)...);
    }

    // 禁止拷贝 (深拷贝开销大，建议显式写 clone 方法或手动拷贝)
    HyresArray(const HyresArray&) = delete;
    HyresArray& operator=(const HyresArray&) = delete;

    // 允许移动 (Move Semantics)
    HyresArray(HyresArray&& other) noexcept { move_from(std::move(other)); }
    HyresArray& operator=(HyresArray&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~HyresArray() { release(); }

    // ==========================================
    // 尺寸与重置
    // ==========================================
    template <typename... Dims, typename = std::enable_if_t<(std::is_integral_v<Dims> && ...)>>
    void resize(Dims... dims) {
        std::vector<std::size_t> vdims{static_cast<std::size_t>(dims)...};
        resize_impl(vdims);
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    const std::vector<std::size_t>& dims() const noexcept { return dims_; }

    void clear() noexcept { release(); }

    // ==========================================
    // Host 访问
    // ==========================================
    T* host_data() noexcept { return h_; }
    const T* host_data() const noexcept { return h_; }

    // 多维索引访问 A(i, j, k)
    // 兼容 Fortran 逻辑: i 变化最快
    template <typename... Idx, typename = std::enable_if_t<(std::is_integral_v<Idx> && ...)>>
    T& operator()(Idx... idx) noexcept {
        return h_[linear_index(static_cast<std::size_t>(idx)...)];
    }

    template <typename... Idx, typename = std::enable_if_t<(std::is_integral_v<Idx> && ...)>>
    const T& operator()(Idx... idx) const noexcept {
        return h_[linear_index(static_cast<std::size_t>(idx)...)];
    }

    // 填充值 (memset 0 或者 std::fill)
    void fill(const T& v) {
        if (!h_ || size_ == 0) return;
        std::fill_n(h_, size_, v);
    }

    // ==========================================
    // Device (GPU) 访问与同步
    // ==========================================
#ifdef HYRES_USE_CUDA
    T* device_data() noexcept { return d_; }
    const T* device_data() const noexcept { return d_; }

    // 检查是否已分配显存
    bool has_device() const noexcept { return d_ != nullptr; }

    // CPU -> GPU (如果显存未分配，自动分配)
    void to_device(cudaStream_t stream = 0) {
        ensure_device_allocated();
        HYRES_CUDA_CHECK(cudaMemcpyAsync(d_, h_, bytes(), cudaMemcpyHostToDevice, stream));
    }

    // GPU -> CPU
    void to_host(cudaStream_t stream = 0) {
        detail::throw_if(d_ == nullptr, "HyresArray::to_host(): device buffer not allocated");
        HYRES_CUDA_CHECK(cudaMemcpyAsync(h_, d_, bytes(), cudaMemcpyDeviceToHost, stream));
    }

    // 手动释放显存 (保留内存)
    void free_device() noexcept {
        if (d_) {
            cudaFree(d_);
            d_ = nullptr;
            d_bytes_ = 0;
        }
    }
#else
    // 非 CUDA 模式下的桩接口，保证代码能编译
    T* device_data() noexcept { return nullptr; }
    void to_device(int = 0) {}
    void to_host(int = 0) {}
    void free_device() {}
#endif

private:
    std::vector<std::size_t> dims_;
    T* h_ = nullptr;
    std::size_t size_ = 0;
    static constexpr std::size_t kAlign = 64;

#ifdef HYRES_USE_CUDA
    T* d_ = nullptr;
    std::size_t d_bytes_ = 0;
#endif

    void release() noexcept {
#ifdef HYRES_USE_CUDA
        free_device();
#endif
        if (h_) {
            detail::aligned_free(h_);
            h_ = nullptr;
        }
        dims_.clear();
        size_ = 0;
    }

    void move_from(HyresArray&& other) noexcept {
        dims_ = std::move(other.dims_);
        h_    = other.h_;
        size_ = other.size_;
        other.h_ = nullptr;
        other.size_ = 0;

#ifdef HYRES_USE_CUDA
        d_ = other.d_;
        d_bytes_ = other.d_bytes_;
        other.d_ = nullptr;
        other.d_bytes_ = 0;
#endif
    }

    void resize_impl(const std::vector<std::size_t>& vdims) {
        detail::throw_if(vdims.empty(), "HyresArray::resize(): dims cannot be empty");
        std::size_t n = 1;
        for (auto d : vdims) {
            n = detail::mul_overflow_safe(n, d);
        }
        dims_ = vdims;
        size_ = n;

        if (h_) {
            detail::aligned_free(h_);
            h_ = nullptr;
        }

        const std::size_t nbytes = bytes();
        h_ = reinterpret_cast<T*>(detail::aligned_malloc(kAlign, nbytes));
        
        // 默认为 0，防止脏数据
        std::memset(h_, 0, nbytes);

#ifdef HYRES_USE_CUDA
        // Resize 后原有显存数据失效，直接释放
        free_device();
#endif
    }

    // 线性索引计算 (i-fastest: i + ni*j + ni*nj*k)
    template <typename... Idx>
    std::size_t linear_index(Idx... idx) const noexcept {
        // 为了性能，Release 模式下不进行边界检查
        // 如果需要调试，可以在这里加 #ifndef NDEBUG 断言
        
        // 展开参数包
        const std::size_t indices[] = {static_cast<std::size_t>(idx)...};
        
        std::size_t offset = indices[0];
        std::size_t stride = dims_[0];

        for (std::size_t k = 1; k < sizeof...(Idx); ++k) {
            offset += indices[k] * stride;
            stride *= dims_[k];
        }
        return offset;
    }

#ifdef HYRES_USE_CUDA
    void ensure_device_allocated() {
        if (size_ == 0 || h_ == nullptr) return;
        const std::size_t need = bytes();
        
        // 如果尚未分配或大小不匹配
        if (d_ == nullptr || d_bytes_ != need) {
            if (d_) cudaFree(d_);
            HYRES_CUDA_CHECK(cudaMalloc((void**)&d_, need));
            d_bytes_ = need;
        }
    }
#endif
};

} // namespace Hyres