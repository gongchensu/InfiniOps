#ifndef INFINI_OPS_CPU_ELEMENTWISE_UNARY_H_
#define INFINI_OPS_CPU_ELEMENTWISE_UNARY_H_

#include "common/generic_utils.h"
#include "cpu/caster_.h"
#include "elementwise/traits.h"
#include "elementwise/unary_functors.h"

namespace infini::ops {

template <typename Op>
class CpuUnaryElementwise : public Op {
 public:
  using Op::Op;

  void operator()(const Tensor input, Tensor out) const override {
    using AllowedDataTypes = typename SupportedDataTypes<Op>::type;

    DataTypeDispatcher<AllowedDataTypes>::template Dispatch<Device::Type::kCpu>(
        this->out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "CpuUnaryElementwise::operator()");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, Tensor out) const {
    if (this->output_size_ == 0) {
      return;
    }

    const auto* input_ptr = static_cast<const T*>(input.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, this->ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < this->output_size_; ++i) {
      const auto input_idx =
          get_idx(i, this->is_input_contiguous_, this->input_shape_.data(),
                  this->input_strides_.data());
      const auto out_idx = get_idx(i, this->is_out_contiguous_,
                                   this->out_shape_.data(),
                                   this->out_strides_.data());

      out_ptr[out_idx] =
          HostUnaryFunctor<Device::Type::kCpu, UnaryModeOf<Op>::value>::template Apply<T>(
              input_ptr[input_idx]);
    }
  }
};

}  // namespace infini::ops

#endif
