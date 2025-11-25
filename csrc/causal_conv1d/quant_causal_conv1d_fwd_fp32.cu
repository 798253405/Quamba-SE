#include "quant_causal_conv1d_fwd_fp32_kernel.cuh"

template void quant_causal_conv1d_fwd_fp32_cuda<int8_t, int8_t>(QuantConvParamsBase &params, cudaStream_t stream);
