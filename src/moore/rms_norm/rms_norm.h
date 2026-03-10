#ifndef INFINI_OPS_MOORE_RMS_NORM_H_
#define INFINI_OPS_MOORE_RMS_NORM_H_

#include <cassert>

#include <musa_runtime_api.h>

#include "base/rms_norm.h"
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

    ScopedDeviceGuard guard(device_index_);

    auto musa_stream = static_cast<musaStream_t>(stream_ ? stream_ : nullptr);
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

    CheckMusa(err, "`LaunchRmsNorm` failed.");
  }

 private:
  static void CheckMusa(musaError_t err, const char* msg) {
    assert((err == musaSuccess) && msg);
  }

  class ScopedDeviceGuard {
   public:
    explicit ScopedDeviceGuard(int target_device) : target_{target_device} {
      if (musaGetDevice(&original_) != musaSuccess) {
        original_ = -1;
      }
      if (target_ >= 0 && target_ != original_) {
        musaSetDevice(target_);
      }
    }

    ~ScopedDeviceGuard() {
      if (original_ >= 0 && target_ >= 0 && original_ != target_) {
        musaSetDevice(original_);
      }
    }

   private:
    int original_{-1};
    int target_{-1};
  };

  int device_index_{0};
};

}  // namespace infini::ops

#endif
