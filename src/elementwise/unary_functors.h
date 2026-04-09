#ifndef INFINI_OPS_ELEMENTWISE_UNARY_FUNCTORS_H_
#define INFINI_OPS_ELEMENTWISE_UNARY_FUNCTORS_H_

#include <cmath>
#include <type_traits>

#include "caster.h"
#include "elementwise/common.h"
#include "elementwise/traits.h"

namespace infini::ops {

template <Device::Type kDev, UnaryMode mode>
struct HostUnaryFunctor {
  template <typename T>
  static T Apply(const T& x) {
    using ComputeType = ElementwiseComputeType<kDev, T>;

    if constexpr (mode == UnaryMode::Abs && std::is_unsigned_v<ComputeType>) {
      return x;
    }

    const auto value = Caster<kDev>::template Cast<ComputeType>(x);

    if constexpr ((mode == UnaryMode::Ceil || mode == UnaryMode::Floor ||
                   mode == UnaryMode::Round) &&
                  std::is_integral_v<ComputeType>) {
      return x;
    } else {
      return Caster<kDev>::template Cast<T>(ApplyCompute(value));
    }
  }

 private:
  template <typename ComputeType>
  static ComputeType ApplyCompute(const ComputeType& value) {
    if constexpr (mode == UnaryMode::Abs) {
      if constexpr (std::is_floating_point_v<ComputeType>) {
        return std::fabs(value);
      } else {
        return static_cast<ComputeType>(std::abs(value));
      }
    } else if constexpr (mode == UnaryMode::Acos) {
      return std::acos(value);
    } else if constexpr (mode == UnaryMode::Acosh) {
      return std::acosh(value);
    } else if constexpr (mode == UnaryMode::Asin) {
      return std::asin(value);
    } else if constexpr (mode == UnaryMode::Asinh) {
      return std::asinh(value);
    } else if constexpr (mode == UnaryMode::Atan) {
      return std::atan(value);
    } else if constexpr (mode == UnaryMode::Atanh) {
      return std::atanh(value);
    } else if constexpr (mode == UnaryMode::Ceil) {
      return std::ceil(value);
    } else if constexpr (mode == UnaryMode::Cos) {
      return std::cos(value);
    } else if constexpr (mode == UnaryMode::Cosh) {
      return std::cosh(value);
    } else if constexpr (mode == UnaryMode::Erf) {
      return std::erf(value);
    } else if constexpr (mode == UnaryMode::Floor) {
      return std::floor(value);
    } else if constexpr (mode == UnaryMode::Log) {
      return std::log(value);
    } else if constexpr (mode == UnaryMode::Neg) {
      return -value;
    } else if constexpr (mode == UnaryMode::Reciprocal) {
      return ComputeType{1} / value;
    } else if constexpr (mode == UnaryMode::Round) {
      return std::nearbyint(value);
    } else if constexpr (mode == UnaryMode::Sign) {
      if constexpr (std::is_unsigned_v<ComputeType>) {
        return value == ComputeType{0} ? ComputeType{0} : ComputeType{1};
      } else {
        return value > ComputeType{0}
                   ? ComputeType{1}
                   : (value < ComputeType{0} ? ComputeType{-1}
                                             : ComputeType{0});
      }
    } else if constexpr (mode == UnaryMode::Sinh) {
      return std::sinh(value);
    } else if constexpr (mode == UnaryMode::Sqrt) {
      return std::sqrt(value);
    } else if constexpr (mode == UnaryMode::Tan) {
      return std::tan(value);
    } else {
      static_assert(mode != mode, "unsupported unary elementwise mode");
    }
  }
};

}  // namespace infini::ops

#endif
