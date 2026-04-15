#ifndef INFINI_OPS_NVIDIA_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_MUL_KERNEL_H_

#include <utility>

#include "cuda/mul/kernel.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kNvidia>
    : public CudaMul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaMul<Runtime<Device::Type::kNvidia>>::CudaMul;
};

}  // namespace infini::ops

#endif
