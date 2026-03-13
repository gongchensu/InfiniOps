#ifndef INFINI_OPS_MOORE_RMS_NORM_H_
#define INFINI_OPS_MOORE_RMS_NORM_H_

#include "base/rms_norm.h"
#include "moore/common.h"
#include "moore/rms_norm/launch.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kMoore> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm{input, weight, eps, out},
        device_index_{input.device().index()} {}

  Operator(const Tensor input, const Tensor weight, Tensor out)
      : Operator{input, weight, 1e-6f, out} {}

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    if (batch_size_ == 0 || nhead_ == 0 || dim_ == 0) {
      return;
    }

    assert(
        out.dtype() == input.dtype() && out.dtype() == weight.dtype() &&
        "Operator `RmsNorm` requires all input and output tensors to have the "
        "same dtype.");

    moore_utils::ScopedDeviceGuard guard(device_index_);

    auto musa_stream = moore_utils::GetMusaStream(stream_);
    auto stride_input_batch = input_strides_.size() > 1 ? input_strides_[0] : 0;
    auto stride_input_nhead =
        input_strides_.size() > 1 ? input_strides_[1] : input_strides_[0];
    auto stride_out_batch = out_strides_.size() > 1 ? out_strides_[0] : 0;
    auto stride_out_nhead =
        out_strides_.size() > 1 ? out_strides_[1] : out_strides_[0];

    auto err = rms_norm::moore::LaunchRmsNorm(
        out.data(), input.data(), weight.data(), stride_out_batch,
        stride_out_nhead, stride_input_batch, stride_input_nhead, batch_size_,
        nhead_, dim_, eps, static_cast<int>(out.dtype()), musa_stream);

    moore_utils::CheckMusa(err, "`LaunchRmsNorm` failed.");
  }

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
