#ifndef INFINI_OPS_BASE_BINARY_ELEMENTWISE_H_
#define INFINI_OPS_BASE_BINARY_ELEMENTWISE_H_

#include <cassert>
#include <type_traits>

#include "elementwise/traits.h"
#include "operator.h"

namespace infini::ops::detail {

class BinaryElementwiseBase {
 public:
  BinaryElementwiseBase(const Tensor input, const Tensor other, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        other_type_{other.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        other_shape_{other.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        other_strides_{other.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_other_contiguous_{other.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(!out.HasBroadcastDim() &&
           "the output of a binary elementwise operator should NOT have a "
           "broadcasted dim");
    assert(input_type_ == other_type_ && other_type_ == out_type_ &&
           "a binary elementwise operator requires all tensors to have the "
           "same dtype");
    assert(input_shape_ == out_shape_ && other_shape_ == out_shape_ &&
           "a binary elementwise operator requires all tensors to have the "
           "same shape");
  }

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType other_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape other_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides other_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_other_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops::detail

#define INFINI_OPS_DECLARE_BINARY_ELEMENTWISE_OP(Name, Mode, DTypeList)     \
  class Name                                                                 \
      : public Operator<Name>, protected detail::BinaryElementwiseBase {     \
   public:                                                                   \
    Name(const Tensor input, const Tensor other, Tensor out)                 \
        : detail::BinaryElementwiseBase(input, other, out) {}                \
                                                                             \
    virtual void operator()(const Tensor input, const Tensor other,          \
                            Tensor out) const = 0;                           \
  };                                                                         \
                                                                             \
  template <>                                                                \
  struct BinaryModeOf<Name>                                                  \
      : std::integral_constant<BinaryMode, BinaryMode::Mode> {};             \
                                                                             \
  template <>                                                                \
  struct SupportedDataTypes<Name> {                                          \
    using type = DTypeList;                                                  \
  };

#endif
