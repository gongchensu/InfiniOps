#ifndef INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_

#include <algorithm>
#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

// clang-format off
#include "hygon/runtime_.h"
// clang-format on

#include "cuda/rms_norm/kernel.h"

namespace infini::ops {

namespace rms_norm {

struct HygonBackend {
  using Stream = typename Runtime<Device::Type::kHygon>::Stream;

  static constexpr Device::Type kDeviceType = Device::Type::kHygon;

  static constexpr int max_block_size = 256;

  static int GetOptimalBlockSize() {
    return std::min(RuntimeUtils<Device::Type::kHygon>::GetOptimalBlockSize(),
                    max_block_size);
  }
};

}  // namespace rms_norm

template <>
class Operator<RmsNorm, Device::Type::kHygon>
    : public CudaRmsNorm<rms_norm::HygonBackend> {
 public:
  using CudaRmsNorm<rms_norm::HygonBackend>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
