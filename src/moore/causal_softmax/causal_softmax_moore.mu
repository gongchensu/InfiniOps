#include <cmath>
#include <type_traits>

#include <cub/block/block_reduce.cuh>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "data_type.h"
#include "moore/causal_softmax/launch.h"

namespace infini::ops::causal_softmax::moore {

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

struct BlockMaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a > b ? a : b;
  }
};

template <unsigned int BLOCK_SIZE, typename Data>
__device__ __forceinline__ float BlockMax(const Data* data_ptr, size_t count) {
  float thread_max = count > 0 ? ToFloat(data_ptr[0]) : 0.0f;
  for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    float value = ToFloat(data_ptr[i]);
    thread_max = value > thread_max ? value : thread_max;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  return BlockReduce(temp_storage).Reduce(thread_max, BlockMaxOp{});
}

template <unsigned int BLOCK_SIZE, typename Data>
__device__ __forceinline__ float BlockSum(const Data* data_ptr, size_t count) {
  float thread_sum = 0.0f;
  for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    thread_sum += ToFloat(data_ptr[i]);
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  return BlockReduce(temp_storage).Sum(thread_sum);
}

template <unsigned int BLOCK_SIZE, typename Data>
__global__ void CausalSoftmaxKernel(
    Data* __restrict__ out_ptr, const Data* __restrict__ input_ptr,
    size_t batch_size, size_t seq_len, size_t total_seq_len,
    ptrdiff_t stride_out_batch, ptrdiff_t stride_out_row,
    ptrdiff_t stride_input_batch, ptrdiff_t stride_input_row) {
  size_t row_idx = blockIdx.x;
  size_t batch_idx = blockIdx.y;

  if (batch_idx >= batch_size || row_idx >= seq_len) {
    return;
  }

  auto* out_row =
      out_ptr + batch_idx * stride_out_batch + row_idx * stride_out_row;
  const auto* input_row =
      input_ptr + batch_idx * stride_input_batch + row_idx * stride_input_row;

  size_t valid_len = total_seq_len - seq_len + row_idx + 1;

  __shared__ float max_val;
  float block_max = BlockMax<BLOCK_SIZE, Data>(input_row, valid_len);
  if (threadIdx.x == 0) {
    max_val = block_max;
  }
  __syncthreads();

  for (size_t col = threadIdx.x; col < total_seq_len; col += BLOCK_SIZE) {
    if (col < valid_len) {
      float diff = ToFloat(input_row[col]) - max_val;
      out_row[col] = FromFloat<Data>(expf(diff));
    } else {
      out_row[col] = FromFloat<Data>(0.0f);
    }
  }
  __syncthreads();

  __shared__ float sum_val;
  float block_sum = BlockSum<BLOCK_SIZE, Data>(out_row, total_seq_len);
  if (threadIdx.x == 0) {
    sum_val = block_sum;
  }
  __syncthreads();

  for (size_t col = threadIdx.x; col < total_seq_len; col += BLOCK_SIZE) {
    out_row[col] = FromFloat<Data>(ToFloat(out_row[col]) / sum_val);
  }
}

template <unsigned int BLOCK_SIZE, typename Data>
musaError_t LaunchKernel(void* out, const void* input, size_t batch_size,
                         size_t seq_len, size_t total_seq_len,
                         ptrdiff_t stride_out_batch, ptrdiff_t stride_out_row,
                         ptrdiff_t stride_input_batch,
                         ptrdiff_t stride_input_row, musaStream_t stream) {
  if (batch_size == 0 || seq_len == 0 || total_seq_len == 0) {
    return musaSuccess;
  }

  dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(batch_size));
  CausalSoftmaxKernel<BLOCK_SIZE, Data><<<grid, BLOCK_SIZE, 0, stream>>>(
      static_cast<Data*>(out), static_cast<const Data*>(input), batch_size,
      seq_len, total_seq_len, stride_out_batch, stride_out_row,
      stride_input_batch, stride_input_row);

  return musaGetLastError();
}

musaError_t LaunchCausalSoftmax(void* out, const void* input, size_t batch_size,
                                size_t seq_len, size_t total_seq_len,
                                ptrdiff_t stride_out_batch,
                                ptrdiff_t stride_out_row,
                                ptrdiff_t stride_input_batch,
                                ptrdiff_t stride_input_row, int dtype,
                                musaStream_t stream) {
  switch (static_cast<DataType>(dtype)) {
    case DataType::kFloat16:
      return LaunchKernel<256, half>(out, input, batch_size, seq_len,
                                     total_seq_len, stride_out_batch,
                                     stride_out_row, stride_input_batch,
                                     stride_input_row, stream);
    case DataType::kBFloat16:
      return LaunchKernel<256, cuda_bfloat16>(
          out, input, batch_size, seq_len, total_seq_len, stride_out_batch,
          stride_out_row, stride_input_batch, stride_input_row, stream);
    case DataType::kFloat32:
      return LaunchKernel<256, float>(
          out, input, batch_size, seq_len, total_seq_len, stride_out_batch,
          stride_out_row, stride_input_batch, stride_input_row, stream);
    default:
      return musaErrorInvalidValue;
  }
}

}  // namespace infini::ops::causal_softmax::moore
