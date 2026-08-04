#ifndef INFINI_OPS_GRAPH_CAPTURE_HANDLE_H_
#define INFINI_OPS_GRAPH_CAPTURE_HANDLE_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "handle.h"

namespace infini::ops {

// The recording owner retains the input storage, metadata and operator until
// the last replay completes. Each segment owns its own pre-launch updates.
class GraphCaptureHandle final : public Handle {
 public:
  using Updates = std::vector<std::function<void()>>;

  GraphCaptureHandle(const void* input, const std::vector<int64_t>& values,
                     Updates& updates)
      : input(input), values(values), updates(updates) {}

  std::unique_ptr<Handle> Clone() const override {
    return std::make_unique<GraphCaptureHandle>(*this);
  }

  const void* const input;
  const std::vector<int64_t>& values;
  Updates& updates;
};

}  // namespace infini::ops

#endif
