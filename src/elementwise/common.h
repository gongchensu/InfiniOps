#ifndef INFINI_OPS_ELEMENTWISE_COMMON_H_
#define INFINI_OPS_ELEMENTWISE_COMMON_H_

#include <type_traits>

#include "data_type.h"

namespace infini::ops {

template <Device::Type kDev, typename T>
using ElementwiseComputeType =
    std::conditional_t<IsFP16<kDev, T> || IsBFloat16<kDev, T>, float, T>;

}  // namespace infini::ops

#endif
