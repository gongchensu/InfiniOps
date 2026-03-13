#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include "base/swiglu.h"
#include "moore/common.h"
#include "moore/swiglu/launch.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kMoore> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu{input, gate, out}, device_index_{input.device().index()} {
    const auto required_workspace_size{workspace_size_in_bytes()};
    if (required_workspace_size == 0) {
      return;
    }

    moore_utils::ScopedDeviceGuard guard(device_index_);
    moore_utils::CheckMusa(
        musaMalloc(reinterpret_cast<void**>(&default_workspace_),
                   required_workspace_size),
        "`musaMalloc` failed for `default_workspace_`.");
    InitializeDefaultWorkspace();
  }

  ~Operator() override {
    moore_utils::ScopedDeviceGuard guard(device_index_);

    moore_utils::FreeDevice(default_workspace_);
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    moore_utils::ScopedDeviceGuard guard(device_index_);

    auto musa_stream = moore_utils::GetMusaStream(stream_);
    auto* workspace{workspace_ ? workspace_ : default_workspace_};
    const auto required_workspace_size{workspace_size_in_bytes()};
    const auto workspace_size{workspace_ && workspace_size_in_bytes_
                                  ? workspace_size_in_bytes_
                                  : required_workspace_size};
    auto layout{WorkspaceLayout{}};

    if (required_workspace_size != 0) {
      assert(workspace != nullptr && "`Swiglu` requires a workspace buffer.");
      assert(workspace_size >= required_workspace_size &&
             "`workspace_size_in_bytes` is insufficient for `Swiglu`.");

      layout = ResolveWorkspaceLayout(workspace);
      if (workspace_ != nullptr) {
        CopyMetadataAsync(layout, musa_stream);
      }
    }

    auto err = swiglu::moore::LaunchSwiglu(
        input.data(), gate.data(), out.data(), layout.out_shape,
        layout.input_shape, layout.gate_shape, layout.out_strides,
        layout.input_strides, layout.gate_strides, output_size_, ndim_,
        is_out_contiguous_, is_input_contiguous_, is_gate_contiguous_,
        static_cast<int>(out_type_), musa_stream);

    moore_utils::CheckMusa(err, "`LaunchSwiglu` failed.");
  }

  std::size_t workspace_size_in_bytes() const override {
    if (output_size_ == 0 || ndim_ == 0) {
      return 0;
    }

    return MetadataWorkspaceSizeInBytes();
  }

 private:
  void InitializeDefaultWorkspace() {
    CopyMetadata(ResolveWorkspaceLayout(default_workspace_));
  }

  void CopyMetadata(const WorkspaceLayout& layout) const {
    const auto shape_size{ndim_ * sizeof(Tensor::Size)};
    const auto strides_size{ndim_ * sizeof(Tensor::Stride)};

    moore_utils::CheckMusa(
        musaMemcpy(layout.input_shape, input_shape_.data(), shape_size,
                   musaMemcpyHostToDevice),
        "`musaMemcpy` failed for `input_shape`.");
    moore_utils::CheckMusa(
        musaMemcpy(layout.gate_shape, gate_shape_.data(), shape_size,
                   musaMemcpyHostToDevice),
        "`musaMemcpy` failed for `gate_shape`.");
    moore_utils::CheckMusa(
        musaMemcpy(layout.out_shape, out_shape_.data(), shape_size,
                   musaMemcpyHostToDevice),
        "`musaMemcpy` failed for `out_shape`.");
    moore_utils::CheckMusa(
        musaMemcpy(layout.input_strides, input_strides_.data(), strides_size,
                   musaMemcpyHostToDevice),
        "`musaMemcpy` failed for `input_strides`.");
    moore_utils::CheckMusa(
        musaMemcpy(layout.gate_strides, gate_strides_.data(), strides_size,
                   musaMemcpyHostToDevice),
        "`musaMemcpy` failed for `gate_strides`.");
    moore_utils::CheckMusa(
        musaMemcpy(layout.out_strides, out_strides_.data(), strides_size,
                   musaMemcpyHostToDevice),
        "`musaMemcpy` failed for `out_strides`.");
  }

  void CopyMetadataAsync(const WorkspaceLayout& layout,
                         musaStream_t musa_stream) const {
    const auto shape_size{ndim_ * sizeof(Tensor::Size)};
    const auto strides_size{ndim_ * sizeof(Tensor::Stride)};

    moore_utils::CheckMusa(
        musaMemcpyAsync(layout.input_shape, input_shape_.data(), shape_size,
                        musaMemcpyHostToDevice, musa_stream),
        "`musaMemcpyAsync` failed for `input_shape`.");
    moore_utils::CheckMusa(
        musaMemcpyAsync(layout.gate_shape, gate_shape_.data(), shape_size,
                        musaMemcpyHostToDevice, musa_stream),
        "`musaMemcpyAsync` failed for `gate_shape`.");
    moore_utils::CheckMusa(
        musaMemcpyAsync(layout.out_shape, out_shape_.data(), shape_size,
                        musaMemcpyHostToDevice, musa_stream),
        "`musaMemcpyAsync` failed for `out_shape`.");
    moore_utils::CheckMusa(
        musaMemcpyAsync(layout.input_strides, input_strides_.data(),
                        strides_size, musaMemcpyHostToDevice, musa_stream),
        "`musaMemcpyAsync` failed for `input_strides`.");
    moore_utils::CheckMusa(
        musaMemcpyAsync(layout.gate_strides, gate_strides_.data(), strides_size,
                        musaMemcpyHostToDevice, musa_stream),
        "`musaMemcpyAsync` failed for `gate_strides`.");
    moore_utils::CheckMusa(
        musaMemcpyAsync(layout.out_strides, out_strides_.data(), strides_size,
                        musaMemcpyHostToDevice, musa_stream),
        "`musaMemcpyAsync` failed for `out_strides`.");
  }

  int device_index_{0};

  void* default_workspace_{nullptr};
};

}  // namespace infini::ops

#endif
