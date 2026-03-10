#include <type_traits>

#include <cub/block/block_reduce.cuh>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "data_type.h"
#include "moore/rms_norm/launch.h"

namespace infini::ops::rms_norm::moore {

using cuda_bfloat16 = mt_bfloat16;

template <typename T>
__device__ __forceinline__ float ToFloat(const T& value) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
    return __bfloat162float(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  if constexpr (std::is_same_v<T, half>) {
    return __float2half(value);
  } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    return static_cast<T>(value);
  }
}

template <unsigned int BLOCK_SIZE, typename Data, typename Compute>
__device__ __forceinline__ Compute SumSquared(const Data* data_ptr,
                                              size_t count) {
  Compute sum = 0;
  for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    const auto value = static_cast<Compute>(ToFloat(data_ptr[i]));
    sum += value * value;
  }

  using BlockReduce = cub::BlockReduce<Compute, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  return BlockReduce(temp_storage).Sum(sum);
}

template <unsigned int BLOCK_SIZE, typename Data>
__global__ void RmsNormKernel(Data* __restrict__ out,
                              ptrdiff_t stride_out_batch,
                              ptrdiff_t stride_out_nhead,
                              const Data* __restrict__ input,
                              ptrdiff_t stride_input_batch,
                              ptrdiff_t stride_input_nhead,
                              const Data* __restrict__ weight, size_t nhead,
                              size_t dim, float epsilon) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto* out_ptr = out + batch_idx * stride_out_batch + head_idx * stride_out_nhead;
  const auto* input_ptr =
      input + batch_idx * stride_input_batch + head_idx * stride_input_nhead;

  float sum = SumSquared<BLOCK_SIZE, Data, float>(input_ptr, dim);

  __shared__ float rms;
  if (threadIdx.x == 0) {
    rms = rsqrtf(sum / static_cast<float>(dim) + epsilon);
  }
  __syncthreads();

  for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
    const float value = ToFloat(input_ptr[i]) * ToFloat(weight[i]) * rms;
    out_ptr[i] = FromFloat<Data>(value);
  }
}

template <unsigned int BLOCK_SIZE, typename Data>
musaError_t LaunchKernel(void* out, const void* input, const void* weight,
                         ptrdiff_t stride_out_batch,
                         ptrdiff_t stride_out_nhead,
                         ptrdiff_t stride_input_batch,
                         ptrdiff_t stride_input_nhead, size_t batch_size,
                         size_t nhead, size_t dim, float epsilon,
                         musaStream_t stream) {
  if (batch_size == 0 || nhead == 0 || dim == 0) {
    return musaSuccess;
  }

  RmsNormKernel<BLOCK_SIZE, Data><<<batch_size * nhead, BLOCK_SIZE, 0, stream>>>(
      static_cast<Data*>(out), stride_out_batch, stride_out_nhead,
      static_cast<const Data*>(input), stride_input_batch, stride_input_nhead,
      static_cast<const Data*>(weight), nhead, dim, epsilon);

  return musaGetLastError();
}

musaError_t LaunchRmsNorm(void* out, const void* input, const void* weight,
                          ptrdiff_t stride_out_batch,
                          ptrdiff_t stride_out_nhead,
                          ptrdiff_t stride_input_batch,
                          ptrdiff_t stride_input_nhead, size_t batch_size,
                          size_t nhead, size_t dim, float epsilon, int dtype,
                          musaStream_t stream) {
  switch (static_cast<DataType>(dtype)) {
    case DataType::kFloat16:
      return LaunchKernel<256, half>(
          out, input, weight, stride_out_batch, stride_out_nhead,
          stride_input_batch, stride_input_nhead, batch_size, nhead, dim,
          epsilon, stream);
    case DataType::kBFloat16:
      return LaunchKernel<256, cuda_bfloat16>(
          out, input, weight, stride_out_batch, stride_out_nhead,
          stride_input_batch, stride_input_nhead, batch_size, nhead, dim,
          epsilon, stream);
    case DataType::kFloat32:
      return LaunchKernel<256, float>(
          out, input, weight, stride_out_batch, stride_out_nhead,
          stride_input_batch, stride_input_nhead, batch_size, nhead, dim,
          epsilon, stream);
    default:
      return musaErrorInvalidValue;
  }
}

}  // namespace infini::ops::rms_norm::moore
