#ifndef INFINI_OPS_ELEMENTWISE_TRAITS_H_
#define INFINI_OPS_ELEMENTWISE_TRAITS_H_

#include <string_view>
#include <utility>

#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

enum class UnaryMode {
  Abs,
  Acos,
  Acosh,
  Asin,
  Asinh,
  Atan,
  Atanh,
  Ceil,
  Cos,
  Cosh,
  Erf,
  Floor,
  Log,
  Neg,
  Reciprocal,
  Round,
  Sign,
  Sinh,
  Sqrt,
  Tan,
};

enum class BinaryMode {
  Div,
  Max,
  Min,
  Mod,
  Pow,
};

template <typename Op>
struct UnaryModeOf;

template <typename Op>
struct BinaryModeOf;

template <typename Op>
struct SupportedDataTypes;

using SignedAndFloatTypes = ConcatType<AllFloatTypes, IntTypes>;

template <typename List>
struct DataTypeDispatcher;

template <DataType... dtypes>
struct DataTypeDispatcher<List<dtypes...>> {
  template <Device::Type kDev, typename Functor, typename... Args>
  static auto Dispatch(DataType dtype, Functor&& func,
                       std::string_view context_str = "", Args&&... args) {
    return DispatchFunc<kDev, dtypes...>(dtype, std::forward<Functor>(func),
                                         context_str,
                                         std::forward<Args>(args)...);
  }
};

}  // namespace infini::ops

#endif
