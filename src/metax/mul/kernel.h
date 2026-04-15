#ifndef INFINI_OPS_METAX_MUL_KERNEL_H_
#define INFINI_OPS_METAX_MUL_KERNEL_H_

#include <utility>

#include "cuda/mul/kernel.h"
#include "metax/caster.cuh"
#include "metax/runtime_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMetax>
    : public CudaMul<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMul<Runtime<Device::Type::kMetax>>::CudaMul;
};

}  // namespace infini::ops

#endif
