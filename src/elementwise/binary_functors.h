#ifndef INFINI_OPS_ELEMENTWISE_BINARY_FUNCTORS_H_
#define INFINI_OPS_ELEMENTWISE_BINARY_FUNCTORS_H_

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "caster.h"
#include "elementwise/common.h"
#include "elementwise/traits.h"

namespace infini::ops {

template <Device::Type kDev, BinaryMode mode>
struct HostBinaryFunctor {
  template <typename T>
  static T Apply(const T& lhs, const T& rhs) {
    using ComputeType = ElementwiseComputeType<kDev, T>;

    const auto lhs_value = Caster<kDev>::template Cast<ComputeType>(lhs);
    const auto rhs_value = Caster<kDev>::template Cast<ComputeType>(rhs);

    return Caster<kDev>::template Cast<T>(ApplyCompute(lhs_value, rhs_value));
  }

 private:
  template <typename ComputeType>
  static ComputeType ApplyCompute(const ComputeType& lhs,
                                  const ComputeType& rhs) {
    if constexpr (mode == BinaryMode::Div) {
      return lhs / rhs;
    } else if constexpr (mode == BinaryMode::Max) {
      if constexpr (std::is_floating_point_v<ComputeType>) {
        return std::fmax(lhs, rhs);
      } else {
        return std::max(lhs, rhs);
      }
    } else if constexpr (mode == BinaryMode::Min) {
      if constexpr (std::is_floating_point_v<ComputeType>) {
        return std::fmin(lhs, rhs);
      } else {
        return std::min(lhs, rhs);
      }
    } else if constexpr (mode == BinaryMode::Mod) {
      return ApplyRemainder(lhs, rhs);
    } else if constexpr (mode == BinaryMode::Pow) {
      return std::pow(lhs, rhs);
    } else {
      static_assert(mode != mode, "unsupported binary elementwise mode");
    }
  }

  template <typename ComputeType>
  static ComputeType ApplyRemainder(const ComputeType& lhs,
                                    const ComputeType& rhs) {
    if constexpr (std::is_floating_point_v<ComputeType>) {
      auto remainder = std::fmod(lhs, rhs);
      if (remainder != ComputeType{0} &&
          ((remainder < ComputeType{0}) != (rhs < ComputeType{0}))) {
        remainder += rhs;
      }
      return remainder;
    } else if constexpr (std::is_signed_v<ComputeType>) {
      auto remainder = lhs % rhs;
      if (remainder != ComputeType{0} &&
          ((remainder < ComputeType{0}) != (rhs < ComputeType{0}))) {
        remainder += rhs;
      }
      return remainder;
    } else {
      return lhs % rhs;
    }
  }
};

}  // namespace infini::ops

#endif
