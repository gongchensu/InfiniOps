#ifndef INFINI_OPS_MOORE_ADD_ADD_H_
#define INFINI_OPS_MOORE_ADD_ADD_H_

#include <cassert>
#include <cstddef>

#include <musa_runtime_api.h>

#include "base/add.h"
#include "moore/add/launch.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kMoore> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, out}, device_index_{input.device().index()} {
    if (ndim_ == 0) {
      return;
    }

    ScopedDeviceGuard guard(device_index_);
    InitializeMetadata();
  }

  ~Operator() override {
    if (d_input_shape_ != nullptr) {
      musaFree(d_input_shape_);
    }
    if (d_other_shape_ != nullptr) {
      musaFree(d_other_shape_);
    }
    if (d_out_shape_ != nullptr) {
      musaFree(d_out_shape_);
    }
    if (d_input_strides_ != nullptr) {
      musaFree(d_input_strides_);
    }
    if (d_other_strides_ != nullptr) {
      musaFree(d_other_strides_);
    }
    if (d_out_strides_ != nullptr) {
      musaFree(d_out_strides_);
    }
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    ScopedDeviceGuard guard(device_index_);

    auto musa_stream = static_cast<musaStream_t>(stream_ ? stream_ : nullptr);
    auto err = add::moore::LaunchAdd(
        input.data(), other.data(), out.data(), d_out_shape_, d_input_shape_,
        d_other_shape_, d_out_strides_, d_input_strides_, d_other_strides_,
        output_size_, ndim_, is_out_contiguous_, is_input_contiguous_,
        is_other_contiguous_, static_cast<int>(out_type_), musa_stream);

    CheckMusa(err, "`LaunchAdd` failed.");
  }

 private:
  static void CheckMusa(musaError_t err, const char* msg) {
    assert((err == musaSuccess) && msg);
  }

  void InitializeMetadata() {
    const auto shape_size = ndim_ * sizeof(Tensor::Size);
    const auto strides_size = ndim_ * sizeof(Tensor::Stride);

    CheckMusa(musaMalloc(reinterpret_cast<void**>(&d_input_shape_), shape_size),
              "`musaMalloc` failed for `d_input_shape_`.");
    CheckMusa(musaMalloc(reinterpret_cast<void**>(&d_other_shape_), shape_size),
              "`musaMalloc` failed for `d_other_shape_`.");
    CheckMusa(musaMalloc(reinterpret_cast<void**>(&d_out_shape_), shape_size),
              "`musaMalloc` failed for `d_out_shape_`.");
    CheckMusa(
        musaMalloc(reinterpret_cast<void**>(&d_input_strides_), strides_size),
        "`musaMalloc` failed for `d_input_strides_`.");
    CheckMusa(
        musaMalloc(reinterpret_cast<void**>(&d_other_strides_), strides_size),
        "`musaMalloc` failed for `d_other_strides_`.");
    CheckMusa(musaMalloc(reinterpret_cast<void**>(&d_out_strides_), strides_size),
              "`musaMalloc` failed for `d_out_strides_`.");

    CheckMusa(musaMemcpy(d_input_shape_, input_shape_.data(), shape_size,
                         musaMemcpyHostToDevice),
              "`musaMemcpy` failed for `d_input_shape_`.");
    CheckMusa(musaMemcpy(d_other_shape_, other_shape_.data(), shape_size,
                         musaMemcpyHostToDevice),
              "`musaMemcpy` failed for `d_other_shape_`.");
    CheckMusa(musaMemcpy(d_out_shape_, out_shape_.data(), shape_size,
                         musaMemcpyHostToDevice),
              "`musaMemcpy` failed for `d_out_shape_`.");
    CheckMusa(musaMemcpy(d_input_strides_, input_strides_.data(), strides_size,
                         musaMemcpyHostToDevice),
              "`musaMemcpy` failed for `d_input_strides_`.");
    CheckMusa(musaMemcpy(d_other_strides_, other_strides_.data(), strides_size,
                         musaMemcpyHostToDevice),
              "`musaMemcpy` failed for `d_other_strides_`.");
    CheckMusa(musaMemcpy(d_out_strides_, out_strides_.data(), strides_size,
                         musaMemcpyHostToDevice),
              "`musaMemcpy` failed for `d_out_strides_`.");
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

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_other_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_other_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
