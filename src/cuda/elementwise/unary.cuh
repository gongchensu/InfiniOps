#ifndef INFINI_OPS_CUDA_ELEMENTWISE_UNARY_CUH_
#define INFINI_OPS_CUDA_ELEMENTWISE_UNARY_CUH_

#include "cuda/elementwise/unary_functors.cuh"
#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <Device::Type kDev, typename T, unsigned int BLOCK_SIZE,
          UnaryMode mode>
__global__ void UnaryElementwiseKernel(
    T* __restrict__ out, const T* __restrict__ input,
    const size_t* __restrict__ out_shape, const size_t* __restrict__ input_shape,
    const ptrdiff_t* __restrict__ out_strides,
    const ptrdiff_t* __restrict__ input_strides, size_t output_size,
    size_t ndim, bool out_contiguous, bool input_contiguous) {
  (void)BLOCK_SIZE;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    const auto out_idx =
        out_contiguous ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    const auto input_idx = input_contiguous
                               ? idx
                               : IndexToOffset(idx, ndim, input_shape,
                                               input_strides);

    out[out_idx] =
        CudaUnaryFunctor<kDev, mode>::template Apply<T>(input[input_idx]);
  }
}

}  // namespace infini::ops

#endif
