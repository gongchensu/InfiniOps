#ifndef INFINI_OPS_NVIDIA_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_SILU_AND_MUL_KERNEL_H_

#include <cstdint>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

namespace detail {

// Map each tile to one row of packed [gate, up]. The 16-byte loads and
// stores cover eight adjacent BF16 elements without per-element division.
struct alignas(16) SiluBf16Pack {
  __nv_bfloat16 values[8];
};

template <unsigned kThreads>
__global__ void SiluAndMulPackedBf16(const SiluBf16Pack* input,
                                     SiluBf16Pack* output, size_t row_packs) {
  const size_t col = size_t(blockIdx.x) * kThreads + threadIdx.x;
  if (col >= row_packs) return;
  const size_t row = blockIdx.y;
  const auto gate = input[row * 2 * row_packs + col];
  const auto up = input[(row * 2 + 1) * row_packs + col];
  SiluBf16Pack result;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float x = __bfloat162float(gate.values[i]);
    const float sigmoid = __frcp_rn(__fadd_rn(1.0f, __expf(-x)));
    result.values[i] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(x, sigmoid), __bfloat162float(up.values[i])));
  }
  output[row * row_packs + col] = result;
}

}  // namespace detail

template <>
class Operator<SiluAndMul, Device::Type::kNvidia>
    : public CudaSiluAndMul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kNvidia>>::CudaSiluAndMul;
  void operator()(const Tensor input, Tensor out) const override {
    using Pack = detail::SiluBf16Pack;
    if (out_type_ == DataType::kBFloat16 && is_input_contiguous_ &&
        is_out_contiguous_ && hidden_size_ >= 1024 && hidden_size_ % 8 == 0 &&
        reinterpret_cast<uintptr_t>(input.data()) % alignof(Pack) == 0 &&
        reinterpret_cast<uintptr_t>(out.data()) % alignof(Pack) == 0) {
      const size_t rows = output_size_ / hidden_size_;
      // Small decode batches use the existing flat kernel. grid.y is 16-bit.
      if (rows >= 16 && rows <= 65535) {
        constexpr unsigned kThreads = 128;
        const size_t row_packs = hidden_size_ / 8;
        const dim3 grid(utils::CeilDiv(row_packs, size_t{kThreads}), rows);
        detail::SiluAndMulPackedBf16<kThreads>
            <<<grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream_)>>>(
                reinterpret_cast<const Pack*>(input.data()),
                reinterpret_cast<Pack*>(out.data()), row_packs);
        return;
      }
    }
    CudaSiluAndMul<Runtime<Device::Type::kNvidia>>::operator()(input, out);
  }

  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
