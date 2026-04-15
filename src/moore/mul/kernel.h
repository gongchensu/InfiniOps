#ifndef INFINI_OPS_MOORE_MUL_KERNEL_H_
#define INFINI_OPS_MOORE_MUL_KERNEL_H_

#include <utility>

// clang-format off
#include "moore/polyfills.cuh"
// clang-format on

#include "cuda/mul/kernel.h"
#include "moore/caster.cuh"
#include "moore/polyfills.cuh"
#include "moore/runtime_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMoore>
    : public CudaMul<Runtime<Device::Type::kMoore>> {
 public:
  using CudaMul<Runtime<Device::Type::kMoore>>::CudaMul;
};

}  // namespace infini::ops

#endif
