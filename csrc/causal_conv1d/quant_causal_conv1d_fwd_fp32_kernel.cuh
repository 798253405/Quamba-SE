/******************************************************************************
 * Mode 2-3: FP32 Output Version (Preserve CUDA internal precision)
 *
 * This kernel is identical to quant_causal_conv1d_fwd_kernel, but:
 * - Outputs FP32 instead of INT8
 * - Skips the quantization step (preserves internal FP32 precision)
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"
#include "common/static_switch.h"

/**************************************************/
/*    Causal_conv1d_fwd_fp32_kernel (Mode 2-3)   */
/**************************************************/

template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_fwd_fp32_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = float;  // FP32 output instead of INT8
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 1);  // Input still INT8
    static constexpr int kNElts = 8;
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;

    // Output stores FP32
    using BlockStoreFloatT = cub::BlockStore<float, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : std::max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreFloatT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void quant_causal_conv1d_fwd_fp32_kernel(QuantConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Scaling factors
    float scale_x = params.scale_x;
    float scale_w = params.scale_w;
    float scale_b = params.scale_b;
    // NOTE: We DON'T use scale_out for quantization (Mode 2-3: preserve FP32)
    float scale_wx = scale_w * scale_x;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store_float = reinterpret_cast<typename Ktraits::BlockStoreFloatT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;

    // Output is FP32
    float *out = reinterpret_cast<float *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;

    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);
    bias_val *= scale_b;

    // Load weights
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) {
        weight_vals[i] = float(weight[i * params.weight_width_stride]) * scale_wx;
    }

    // Convolution state
    float state_vals[kWidth] = {0};

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), params.seqlen - chunk * kChunkSize);
        }
        x += kChunkSize;
        __syncthreads();
        if constexpr(!kIsVecLoad) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }
        __syncthreads();
        if constexpr(!kIsVecLoad) { reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1]; }

        // Dequantize input to FP32
        float x_vals[2 * kNElts];
        #pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i) {
            x_vals[i] = float(x_vals_load[i]) * scale_x;
        }

        // Convolution computation (FP32)
        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
            }
        }

        // Update state
        state_vals[kWidth - 1] = x_vals[2 * kNElts - 1];
        #pragma unroll
        for (int w = kWidth - 1; w >= 1; --w) {
            state_vals[w - 1] = state_vals[w];
        }

        // Apply SiLU activation (FP32)
        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        // ===================================================================
        // Mode 2-3: PRESERVE FP32 PRECISION (No quantization!)
        // ===================================================================
        // Original kernel quantizes here:
        //   int tmp = int(roundf(out_vals[i] / scale_out));
        //   out_vals_store[i] = clamp(tmp, -128, 127);
        //
        // Mode 2-3: Directly store FP32 values (preserve internal precision)
        // ===================================================================

        float out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals_store[i] = out_vals[i];  // No quantization!
        }

        // Store FP32 output
        if constexpr(kIsVecLoad) {
            // For vectorized load, we still use regular store for FP32
            Ktraits::BlockStoreFloatT(smem_store_float).Store(out, out_vals_store, params.seqlen - chunk * kChunkSize);
        } else {
            Ktraits::BlockStoreFloatT(smem_store_float).Store(out, out_vals_store, params.seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;
    }
}

template<typename input_t, typename weight_t>
void quant_causal_conv1d_fwd_fp32_cuda(QuantConvParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % 8 == 0, kIsVecLoad, [&] {
        BOOL_SWITCH(params.width == 2, kWidth2, [&] {
            BOOL_SWITCH(params.width == 3, kWidth3, [&] {
                BOOL_SWITCH(params.width == 4, kWidth4, [&] {
                    constexpr int kWidth = !kWidth2 ? (!kWidth3 ? (!kWidth4 ? -1 : 4) : 3) : 2;
                    constexpr int kNThreads = 128;
                    if constexpr (kWidth >= 2 && kWidth <= 4) {
                        using Ktraits = Causal_conv1d_fwd_fp32_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
                        constexpr int kSmemSize = Ktraits::kSmemSize;
                        dim3 grid(params.batch, params.dim);
                        auto kernel = &quant_causal_conv1d_fwd_fp32_kernel<Ktraits>;
                        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                });
            });
        });
    });
}
