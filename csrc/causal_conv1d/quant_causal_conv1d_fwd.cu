#include "quant_causal_conv1d_fwd_kernel.cuh"

template void quant_causal_conv1d_fwd_cuda<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);

template void quant_causal_conv1d_channellast_fwd_cuda<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);

// Mode 5: FP32 output
template void quant_causal_conv1d_fwd_cuda_mode5<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);

// Mode 6: FP32 output (same as Mode 5, used for dual-path comparison)
template void quant_causal_conv1d_fwd_cuda_mode6<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);