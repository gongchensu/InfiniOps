#ifndef INFINI_OPS_BASE_UNARY_ELEMENTWISE_H_
#define INFINI_OPS_BASE_UNARY_ELEMENTWISE_H_

#include <cassert>
#include <type_traits>

#include "elementwise/traits.h"
#include "operator.h"

namespace infini::ops::detail {

class UnaryElementwiseBase {
 public:
  UnaryElementwiseBase(const Tensor input, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(!out.HasBroadcastDim() &&
           "the output of a unary elementwise operator should NOT have a "
           "broadcasted dim");
    assert(input_type_ == out_type_ &&
           "a unary elementwise operator requires the input and output "
           "tensors to have the same dtype");
    assert(input_shape_ == out_shape_ &&
           "a unary elementwise operator requires the input and output "
           "tensors to have the same shape");
  }

 protected:
  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops::detail

#define INFINI_OPS_DECLARE_UNARY_ELEMENTWISE_OP(Name, Mode, DTypeList)     \
  class Name : public Operator<Name>, protected detail::UnaryElementwiseBase { \
   public:                                                                 \
    Name(const Tensor input, Tensor out)                                   \
        : detail::UnaryElementwiseBase(input, out) {}                      \
                                                                           \
    virtual void operator()(const Tensor input, Tensor out) const = 0;     \
  };                                                                       \
                                                                           \
  template <>                                                              \
  struct UnaryModeOf<Name>                                                 \
      : std::integral_constant<UnaryMode, UnaryMode::Mode> {};             \
                                                                           \
  template <>                                                              \
  struct SupportedDataTypes<Name> {                                        \
    using type = DTypeList;                                                \
  };

#endif
