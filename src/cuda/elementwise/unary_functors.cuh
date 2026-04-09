#ifndef INFINI_OPS_CUDA_ELEMENTWISE_UNARY_FUNCTORS_CUH_
#define INFINI_OPS_CUDA_ELEMENTWISE_UNARY_FUNCTORS_CUH_

#include <cmath>
#include <type_traits>

#include "caster.h"
#include "elementwise/common.h"
#include "elementwise/traits.h"

namespace infini::ops {

template <Device::Type kDev, UnaryMode mode>
struct CudaUnaryFunctor {
  template <typename T>
  __device__ __forceinline__ static T Apply(const T& x) {
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
  __device__ __forceinline__ static ComputeType ApplyCompute(
      const ComputeType& value) {
    if constexpr (mode == UnaryMode::Abs) {
      if constexpr (std::is_floating_point_v<ComputeType>) {
        return fabs(value);
      } else {
        return value >= ComputeType{0} ? value : -value;
      }
    } else if constexpr (mode == UnaryMode::Acos) {
      return acos(value);
    } else if constexpr (mode == UnaryMode::Acosh) {
      return acosh(value);
    } else if constexpr (mode == UnaryMode::Asin) {
      return asin(value);
    } else if constexpr (mode == UnaryMode::Asinh) {
      return asinh(value);
    } else if constexpr (mode == UnaryMode::Atan) {
      return atan(value);
    } else if constexpr (mode == UnaryMode::Atanh) {
      return atanh(value);
    } else if constexpr (mode == UnaryMode::Ceil) {
      return ceil(value);
    } else if constexpr (mode == UnaryMode::Cos) {
      return cos(value);
    } else if constexpr (mode == UnaryMode::Cosh) {
      return cosh(value);
    } else if constexpr (mode == UnaryMode::Erf) {
      return erf(value);
    } else if constexpr (mode == UnaryMode::Floor) {
      return floor(value);
    } else if constexpr (mode == UnaryMode::Log) {
      return log(value);
    } else if constexpr (mode == UnaryMode::Neg) {
      return -value;
    } else if constexpr (mode == UnaryMode::Reciprocal) {
      return ComputeType{1} / value;
    } else if constexpr (mode == UnaryMode::Round) {
      return nearbyint(value);
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
      return sinh(value);
    } else if constexpr (mode == UnaryMode::Sqrt) {
      return sqrt(value);
    } else if constexpr (mode == UnaryMode::Tan) {
      return tan(value);
    } else {
      static_assert(mode != mode, "unsupported unary elementwise mode");
    }
  }
};

}  // namespace infini::ops

#endif
