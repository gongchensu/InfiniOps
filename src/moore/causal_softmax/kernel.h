#ifndef INFINI_OPS_MOORE_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_MOORE_CAUSAL_SOFTMAX_KERNEL_H_

#include "base/causal_softmax.h"
#include "moore/common.h"
#include "moore/causal_softmax/launch.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kMoore> : public CausalSoftmax {
 public:
  Operator(const Tensor input, Tensor out)
      : CausalSoftmax{input, out}, device_index_{input.device().index()} {}

  void operator()(const Tensor input, Tensor out) const override {
    if (batch_size_ == 0 || seq_len_ == 0 || total_seq_len_ == 0) {
      return;
    }

    assert(input.dtype() == out.dtype() &&
           "Operator `CausalSoftmax` requires `input` and `out` to have the "
           "same dtype.");

    moore_utils::ScopedDeviceGuard guard(device_index_);

    auto musa_stream = moore_utils::GetMusaStream(stream_);
    auto stride_input_batch = ndim_ == 3 ? input_strides_[0] : 0;
    auto stride_input_row = input_strides_[ndim_ - 2];
    auto stride_out_batch = ndim_ == 3 ? out_strides_[0] : 0;
    auto stride_out_row = out_strides_[ndim_ - 2];

    auto err = causal_softmax::moore::LaunchCausalSoftmax(
        out.data(), input.data(), batch_size_, seq_len_, total_seq_len_,
        stride_out_batch, stride_out_row, stride_input_batch, stride_input_row,
        static_cast<int>(out.dtype()), musa_stream);

    moore_utils::CheckMusa(err, "`LaunchCausalSoftmax` failed.");
  }

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
