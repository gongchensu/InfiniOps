#ifndef INFINI_OPS_HYGON_SWIGLU_KERNEL_H_
#define INFINI_OPS_HYGON_SWIGLU_KERNEL_H_

#include "cuda/swiglu/kernel.h"
#include "hygon/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kHygon>
    : public CudaSwiglu<Runtime<Device::Type::kHygon>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kHygon>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
