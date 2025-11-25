/*
The code is modfied from
https://github.com/state-spaces/mamba
*/

// Split into multiple files to compile in paralell
#include "quant_sscan_fwd_kernel.cuh"

// quant_sscan_fwd_cuda<input_t, weight_t>(params, stream)
template void quant_sscan_fwd_cuda<int8_t, int8_t>(QuantSSMParams &params, cudaStream_t stream);

// Mode 5: FP32 input u
template void quant_sscan_fwd_cuda_mode5<int8_t, int8_t>(QuantSSMParams &params, cudaStream_t stream);

// Mode 6: FP32 input u (same as Mode 5, for dual-path comparison)
template void quant_sscan_fwd_cuda_mode6<int8_t, int8_t>(QuantSSMParams &params, cudaStream_t stream);