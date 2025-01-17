#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "cute/config.hpp"
#include "cute/pointer.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <cassert>
#include <cstdint>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/bfloat16.h>
#include <sys/stat.h>
#include <thrust/host_vector.h>
#include <torch/extension.h>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 750))
#include "cute/arch/copy_sm75.hpp"
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 890))
#include "cute/arch/copy_sm80.hpp"
#else
#include "cute/arch/copy_sm90.hpp"
#endif
#pragma once

namespace mase_cuda {
namespace mxint8 {
namespace dequantize {
template <class TypeX, class TypeScale>
__host__ void dequantize1d_host(TypeX const *x, const int M, TypeScale const *scales, const int group_size,
                                cutlass::bfloat16_t *y) {
    assert(M % group_size == 0);

    const int num_groups = M / group_size;
    int8_t const *x_raw_int8 = reinterpret_cast<int8_t const *>(x);
    uint8_t const *scales_raw_uint8 = reinterpret_cast<uint8_t const *>(scales);
    thrust::host_vector<int8_t> hX(x_raw_int8, x_raw_int8 + M);
    thrust::host_vector<uint8_t> hScales(scales_raw_uint8, scales_raw_uint8 + num_groups);

    for (int i = 0; i < M; ++i) {
        auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8;
        auto exp = static_cast<uint16_t>(hScales[i / group_size]) << 7;
        auto mantissa_abs = abs(hX[i]);
        auto frac = static_cast<uint16_t>((mantissa_abs & 0x3F) << 1);
        auto out = cutlass::bfloat16_t::bitcast(sign | exp | frac);
        auto dont_need_abs = bool(mantissa_abs & 0x40);
        auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0));
        y[i] = dont_need_abs ? out : out - bias;
    }
}

template <class TypeX,                              // input type
          class ShapeX,                             // input/output shape
          class StrideX,                            // input/output stride
          class TypeScale,                          // scale type
          class ScaleShape,                         // scale shape
          class StrideScale,                        // scale stride
          class GroupTiler,                         // group size
          class CtaTiler,                           // CTA tiler, (BLK_M, BLK_K)
          class SmemLayoutX, class SmemLayoutScale, // shared mem layout
          class ThreadLayoutX>
__global__ static void dequantize1d_device(TypeX const *x, ShapeX shape_x, StrideX stride_x, // input
                                           TypeScale const *scales, ScaleShape shape_scale, StrideScale stride_scale,
                                           GroupTiler group_tiler, // scale
                                           CtaTiler cta_tiler, SmemLayoutX layout_sX, SmemLayoutScale layout_sScale,
                                           ThreadLayoutX layout_tX, // cuda
                                           cutlass::bfloat16_t *y) {
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(shape_x) == Int<1>{});
    CUTE_STATIC_ASSERT_V(rank(shape_scale) == Int<1>{});
    CUTE_STATIC_ASSERT_V(rank(group_tiler) == Int<1>{});
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<2>{});

    static_assert(is_static<ThreadLayoutX>::value);
    static_assert(is_static<SmemLayoutX>::value);
    static_assert(is_static<SmemLayoutScale>::value);

    int8_t const *x_raw_int8 = reinterpret_cast<int8_t const *>(x);
    uint8_t const *scales_raw_uint8 = reinterpret_cast<uint8_t const *>(scales);
    Tensor mX_raw = make_tensor(make_gmem_ptr(x_raw_int8), shape_x, stride_x);
    Tensor mX = flatten(flat_divide(mX_raw, group_tiler)); // (_group_size, num_groups):(_1, _group_size)
    Tensor vScale = make_tensor(make_gmem_ptr(scales_raw_uint8), shape_scale, stride_scale); // (num_groups,)
    Tensor mY = make_tensor(make_gmem_ptr(y), shape(mX), stride(mX)); // (_group_size, num_groups):(_1, _group_size)

    // predicate
    auto m_max_coord = size<0>(mX) - blockIdx.x * size<0>(cta_tiler);
    auto k_max_coord = size<1>(mX) - blockIdx.y * size<1>(cta_tiler);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y);
    Tensor gX = local_tile(mX, cta_tiler, cta_coord);                               // (BLK_M, BLK_K)
    Tensor gScale = local_tile(vScale, select<1>(cta_tiler), select<1>(cta_coord)); // (BLK_K,)
    Tensor gY = local_tile(mY, cta_tiler, cta_coord);                               // (BLK_M, BLK_K)

    __shared__ int8_t smemX[size(cta_tiler)];
    __shared__ uint8_t smemScale[size(select<1>(cta_tiler))];

    Tensor sX = make_tensor(make_smem_ptr(smemX), cta_tiler);                    // (BLK_M, BLK_K)
    Tensor sScale = make_tensor(make_smem_ptr(smemScale), select<1>(cta_tiler)); // (BLK_K,)

    // partition copy
    Tensor tXgX = local_partition(gX, layout_tX, threadIdx.x); // (thd_m, thd_k)
    Tensor tXsX = local_partition(sX, layout_sX, threadIdx.x); // (thd_m, thd_k)

    Tensor tXgY = local_partition(gY, layout_tX, threadIdx.x); // (thd_m, thd_k)
    Tensor tXrY = make_tensor_like(tXgY);                      // (thd_m, thd_k)

    // predicate
    Tensor cX = make_identity_tensor(shape(sX));               // (BLK_M, BLK_K)
    Tensor tXcX = local_partition(cX, layout_sX, threadIdx.x); // (thd_m, thd_k)
    Tensor tXpX = make_tensor<bool>(shape(tXgX));              // (thd_m, thd_k)

    CUTE_UNROLL
    for (int i = 0; i < size(tXsX); ++i) {
        tXpX[i] = elem_less(tXcX[i], make_coord(m_max_coord, k_max_coord));
    }

    clear(tXrY);

    // copy
    // copy Scale
    // if (threadIdx.x < size(select<1>(cta_tiler))) {
    //     sScale[threadIdx.x] = gScale[threadIdx.x];
    // }
    if (threadIdx.x < min(size<1>(cta_tiler), k_max_coord)) {
        sScale[threadIdx.x] = gScale[threadIdx.x];
    }
    // copy X
    // CUTE_UNROLL
    // for (int i = 0; i < size(tXgX); ++i) {
    //     tXsX[i] = tXgX[i];
    // }
    copy_if(tXpX, tXgX, tXsX);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // dequantize
    CUTE_UNROLL
    for (int i = 0; i < size(tXrY); ++i) {
        auto scaleIdx = threadIdx.x / size<0>(layout_tX);
        auto exp = static_cast<uint16_t>(sScale[scaleIdx]) << 7;
        auto sign = static_cast<uint16_t>(tXsX[i] & 0x80) << 8;
        auto mantissa_abs = abs(tXsX[i]);
        auto frac = static_cast<uint16_t>((mantissa_abs & 0x3F) << 1);
        auto out = cutlass::bfloat16_t::bitcast(sign | exp | frac);
        auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0));
        auto dont_need_bias = bool(mantissa_abs & 0x40);
        tXrY[i] = dont_need_bias ? out : out - bias;
    }

    // copy back
    // CUTE_UNROLL
    // for (int i = 0; i < size(tXrY); ++i) {
    //     tXgY[i] = tXrY[i];
    // }
    copy_if(tXpX, tXrY, tXgY);
}

torch::Tensor dequantize1d(torch::Tensor x, torch::Tensor scales, const int group_size) {
    using namespace cute;

    if (group_size <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }
    const int m = x.numel() * x.itemsize() / sizeof(uint8_t);
    const int num_groups = m / group_size;
    if (m % group_size != 0) {
        throw std::invalid_argument("m must be divisible by group_size");
    }
    if (x.device() != scales.device()) {
        throw std::invalid_argument("x and scales must be on the same device");
    }

    auto y_options = torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16);
    auto y = torch::empty({m}, y_options);

    auto _x = x.contiguous();
    auto _scales = scales.contiguous();

    auto x_ptr = _x.const_data_ptr();
    auto scales_ptr = _scales.const_data_ptr();
    cutlass::bfloat16_t *y_ptr = reinterpret_cast<cutlass::bfloat16_t *>(y.data_ptr());

    if (_x.device().is_cpu()) {
        dequantize1d_host(x_ptr, m, scales_ptr, group_size, y_ptr);
    } else if (_x.device().is_cuda()) {
        auto shape_x = make_shape(m);
        auto stride_x = make_stride(Int<1>{});
        auto shape_scale = make_shape(num_groups);
        auto stride_scale = make_stride(Int<1>{});
        auto group_tiler = make_shape(group_size);
        auto device = x.get_device();
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

        if (group_size <= 8) {
            auto BLK_M = Int<8>{};
            auto BLK_K = Int<128>{};
            auto thd_m = BLK_M;
            auto thd_k = BLK_K;
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(thd_m, thd_k));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
            dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                                  stride_scale, group_tiler, cta_tiler, layout_sX,
                                                                  layout_sScale, layout_tX, y_ptr);
        } else if (group_size <= 16) {
            auto BLK_M = Int<16>{};
            auto BLK_K = Int<64>{};
            auto thd_m = BLK_M;
            auto thd_k = BLK_K;
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(thd_m, thd_k));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
            dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                                  stride_scale, group_tiler, cta_tiler, layout_sX,
                                                                  layout_sScale, layout_tX, y_ptr);
        } else if (group_size <= 32) {
            auto BLK_M = Int<32>{};
            auto BLK_K = Int<32>{};
            auto thd_m = BLK_M;
            auto thd_k = BLK_K;
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(thd_m, thd_k));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
            dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                                  stride_scale, group_tiler, cta_tiler, layout_sX,
                                                                  layout_sScale, layout_tX, y_ptr);
        } else if (group_size <= 64) {
            auto BLK_M = Int<64>{};
            auto BLK_K = Int<16>{};
            auto thd_m = BLK_M;
            auto thd_k = BLK_K;
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(thd_m, thd_k));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
            dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                                  stride_scale, group_tiler, cta_tiler, layout_sX,
                                                                  layout_sScale, layout_tX, y_ptr);
        } else if (group_size <= 128) {
            auto BLK_M = Int<128>{};
            auto BLK_K = Int<8>{};
            auto thd_m = BLK_M;
            auto thd_k = BLK_K;
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(thd_m, thd_k));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
            dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                                  stride_scale, group_tiler, cta_tiler, layout_sX,
                                                                  layout_sScale, layout_tX, y_ptr);
        } else {
            // larger group sizes
            auto BLK_M = Int<128>{};
            auto BLK_K = Int<8>{};
            auto thd_m = BLK_M;
            auto thd_k = BLK_K;
            auto cta_tiler = make_shape(BLK_M, BLK_K);
            auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
            auto layout_sScale = make_layout(make_shape(BLK_K));
            auto layout_tX = make_layout(make_shape(thd_m, thd_k));
            dim3 dimBlock(size(layout_tX));
            dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
            dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
                                                                  stride_scale, group_tiler, cta_tiler, layout_sX,
                                                                  layout_sScale, layout_tX, y_ptr);
        }
    } else {
        throw std::invalid_argument("x must be on CPU or CUDA");
    }

    y = y.reshape_as(x);
    return y;
}
} // namespace dequantize
} // namespace mxint8
} // namespace mase_cuda