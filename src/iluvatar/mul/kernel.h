#ifndef INFINI_OPS_ILUVATAR_MUL_KERNEL_H_
#define INFINI_OPS_ILUVATAR_MUL_KERNEL_H_

#include <utility>

#include "cuda/mul/kernel.h"
#include "iluvatar/caster.cuh"
#include "iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kIluvatar>
    : public CudaMul<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaMul<Runtime<Device::Type::kIluvatar>>::CudaMul;
};

}  // namespace infini::ops

#endif
