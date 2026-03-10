#include <cstdint>
#include <type_traits>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "data_type.h"
#include "moore/add/launch.h"

namespace infini::ops::add::moore {

using cuda_bfloat16 = mt_bfloat16;

__device__ __forceinline__ size_t IndexToOffset(size_t flat_index, size_t ndim,
                                                const size_t* shape,
                                                const ptrdiff_t* strides) {
  size_t offset = 0;
  for (size_t i = ndim; i-- > 0;) {
    offset += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }

  return offset;
}

template <typename T>
__device__ __forceinline__ T AddValue(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, half>) {
    return __hadd(a, b);
  } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
    return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
  } else if constexpr (std::is_same_v<T, float>) {
    return __fadd_rn(a, b);
  } else {
    return a + b;
  }
}

template <typename T, unsigned int BLOCK_SIZE>
__global__ void AddKernel(
    T* out, const T* input, const T* other, const size_t* out_shape,
    const size_t* input_shape, const size_t* other_shape,
    const ptrdiff_t* out_strides, const ptrdiff_t* input_strides,
    const ptrdiff_t* other_strides, size_t output_size, size_t ndim,
    size_t offset, bool out_contiguous, bool input_contiguous,
    bool other_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
  if (idx >= output_size) {
    return;
  }

  size_t out_idx = out_contiguous ? idx
                                  : IndexToOffset(idx, ndim, out_shape,
                                                  out_strides);
  size_t input_idx = input_contiguous ? idx
                                      : IndexToOffset(idx, ndim, input_shape,
                                                      input_strides);
  size_t other_idx = other_contiguous ? idx
                                      : IndexToOffset(idx, ndim, other_shape,
                                                      other_strides);

  out[out_idx] = AddValue(input[input_idx], other[other_idx]);
}

template <unsigned int BLOCK_SIZE, typename T>
musaError_t LaunchKernel(
    const void* input, const void* other, void* out, const size_t* out_shape,
    const size_t* input_shape, const size_t* other_shape,
    const ptrdiff_t* out_strides, const ptrdiff_t* input_strides,
    const ptrdiff_t* other_strides, size_t output_size, size_t ndim,
    bool out_contiguous, bool input_contiguous, bool other_contiguous,
    musaStream_t stream) {
  if (output_size == 0) {
    return musaSuccess;
  }

  dim3 block(BLOCK_SIZE);
  dim3 grid((output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
  size_t step = static_cast<size_t>(grid.x) * block.x;

  for (size_t offset = 0; offset < output_size; offset += step) {
    AddKernel<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        static_cast<T*>(out), static_cast<const T*>(input),
        static_cast<const T*>(other), out_shape, input_shape, other_shape,
        out_strides, input_strides, other_strides, output_size, ndim, offset,
        out_contiguous, input_contiguous, other_contiguous);
    auto err = musaGetLastError();
    if (err != musaSuccess) {
      return err;
    }
  }

  return musaSuccess;
}

musaError_t LaunchAdd(const void* input, const void* other, void* out,
                      const size_t* out_shape, const size_t* input_shape,
                      const size_t* other_shape, const ptrdiff_t* out_strides,
                      const ptrdiff_t* input_strides,
                      const ptrdiff_t* other_strides, size_t output_size,
                      size_t ndim, bool out_contiguous, bool input_contiguous,
                      bool other_contiguous, int dtype, musaStream_t stream) {
  switch (static_cast<DataType>(dtype)) {
    case DataType::kInt8:
      return LaunchKernel<256, std::int8_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kInt16:
      return LaunchKernel<256, std::int16_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kInt32:
      return LaunchKernel<256, std::int32_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kInt64:
      return LaunchKernel<256, std::int64_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kUInt8:
      return LaunchKernel<256, std::uint8_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kUInt16:
      return LaunchKernel<256, std::uint16_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kUInt32:
      return LaunchKernel<256, std::uint32_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kUInt64:
      return LaunchKernel<256, std::uint64_t>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kFloat16:
      return LaunchKernel<256, half>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kBFloat16:
      return LaunchKernel<256, cuda_bfloat16>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kFloat32:
      return LaunchKernel<256, float>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    case DataType::kFloat64:
      return LaunchKernel<256, double>(
          input, other, out, out_shape, input_shape, other_shape, out_strides,
          input_strides, other_strides, output_size, ndim, out_contiguous,
          input_contiguous, other_contiguous, stream);
    default:
      return musaErrorInvalidValue;
  }
}

}  // namespace infini::ops::add::moore
