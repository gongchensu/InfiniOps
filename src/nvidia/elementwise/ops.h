#ifndef INFINI_OPS_NVIDIA_ELEMENTWISE_OPS_H_
#define INFINI_OPS_NVIDIA_ELEMENTWISE_OPS_H_

#include "base/abs.h"
#include "base/acos.h"
#include "base/acosh.h"
#include "base/asin.h"
#include "base/asinh.h"
#include "base/atan.h"
#include "base/atanh.h"
#include "base/ceil.h"
#include "base/cos.h"
#include "base/cosh.h"
#include "base/div.h"
#include "base/erf.h"
#include "base/floor.h"
#include "base/log.h"
#include "base/max.h"
#include "base/min.h"
#include "base/mod.h"
#include "base/neg.h"
#include "base/pow.h"
#include "base/reciprocal.h"
#include "base/round.h"
#include "base/sign.h"
#include "base/sinh.h"
#include "base/sqrt.h"
#include "base/tan.h"
#include "cuda/elementwise/binary.h"
#include "cuda/elementwise/unary.h"
#include "nvidia/caster.cuh"
#include "nvidia/runtime_.h"

namespace infini::ops {

#define INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(OpName)                             \
  template <>                                                                 \
  class Operator<OpName, Device::Type::kNvidia>                               \
      : public CudaUnaryElementwise<Runtime<Device::Type::kNvidia>, OpName> { \
   public:                                                                    \
    using CudaUnaryElementwise<Runtime<Device::Type::kNvidia>,                \
                               OpName>::CudaUnaryElementwise;                 \
  };

#define INFINI_OPS_DEFINE_NVIDIA_BINARY_OP(OpName)                            \
  template <>                                                                 \
  class Operator<OpName, Device::Type::kNvidia>                               \
      : public CudaBinaryElementwise<Runtime<Device::Type::kNvidia>, OpName> { \
   public:                                                                    \
    using CudaBinaryElementwise<Runtime<Device::Type::kNvidia>,               \
                                OpName>::CudaBinaryElementwise;               \
  };

INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Abs)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Acos)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Acosh)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Asin)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Asinh)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Atan)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Atanh)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Ceil)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Cos)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Cosh)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Erf)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Floor)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Log)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Neg)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Reciprocal)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Round)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Sign)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Sinh)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Sqrt)
INFINI_OPS_DEFINE_NVIDIA_UNARY_OP(Tan)

INFINI_OPS_DEFINE_NVIDIA_BINARY_OP(Div)
INFINI_OPS_DEFINE_NVIDIA_BINARY_OP(Max)
INFINI_OPS_DEFINE_NVIDIA_BINARY_OP(Min)
INFINI_OPS_DEFINE_NVIDIA_BINARY_OP(Mod)
INFINI_OPS_DEFINE_NVIDIA_BINARY_OP(Pow)

#undef INFINI_OPS_DEFINE_NVIDIA_UNARY_OP
#undef INFINI_OPS_DEFINE_NVIDIA_BINARY_OP

}  // namespace infini::ops

#endif
