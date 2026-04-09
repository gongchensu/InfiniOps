#ifndef INFINI_OPS_CUDA_ELEMENTWISE_BINARY_H_
#define INFINI_OPS_CUDA_ELEMENTWISE_BINARY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/generic_utils.h"
#include "cuda/elementwise/binary.cuh"
#include "cuda/runtime_utils.h"
#include "elementwise/traits.h"

namespace infini::ops {

template <typename Backend, typename Op>
class CudaBinaryElementwise : public Op {
 public:
  using Op::Op;

  CudaBinaryElementwise(const Tensor input, const Tensor other, Tensor out)
      : Op{input, other, out} {
    if (this->ndim_ == 0) {
      return;
    }

    const auto shape_size = this->ndim_ * sizeof(*d_input_shape_);
    const auto strides_size = this->ndim_ * sizeof(*d_input_strides_);
    const auto metadata_size = 3 * (shape_size + strides_size);

    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc(reinterpret_cast<void**>(&d_metadata_), metadata_size);

    size_t offset = 0;
    d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, this->input_shape_.data(), shape_size);
    offset += shape_size;

    d_other_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, this->other_shape_.data(), shape_size);
    offset += shape_size;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, this->out_shape_.data(), shape_size);
    offset += shape_size;

    d_input_strides_ =
        reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, this->input_strides_.data(),
                strides_size);
    offset += strides_size;

    d_other_strides_ =
        reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, this->other_strides_.data(),
                strides_size);
    offset += strides_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, this->out_strides_.data(),
                strides_size);

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::MemcpyHostToDevice);
  }

  ~CudaBinaryElementwise() {
    if (d_metadata_ != nullptr) {
      Backend::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    if (this->output_size_ == 0) {
      return;
    }

    const int block_size =
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
    using AllowedDataTypes = typename SupportedDataTypes<Op>::type;

    DispatchFunc<AllowedDataTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(this->out_type_), block_size},
        [&](auto list_tag) {
          using T =
              TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(this->stream_
                                                        ? this->stream_
                                                        : 0);
          dim3 block_dims(
              std::min(static_cast<Tensor::Size>(block_size), this->output_size_));
          dim3 grid_dims(utils::CeilDiv(this->output_size_, block_dims.x));

          BinaryElementwiseKernel<Backend::kDeviceType, T, kBlockSize,
                                  BinaryModeOf<Op>::value>
              <<<grid_dims, block_dims, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()),
                  reinterpret_cast<const T*>(other.data()), d_out_shape_,
                  d_input_shape_, d_other_shape_, d_out_strides_,
                  d_input_strides_, d_other_strides_, this->output_size_,
                  this->ndim_, this->is_out_contiguous_,
                  this->is_input_contiguous_, this->is_other_contiguous_);
        },
        "CudaBinaryElementwise::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_other_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_other_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
