#ifndef INFINI_OPS_MOORE_COMMON_H_
#define INFINI_OPS_MOORE_COMMON_H_

#include <cassert>

#include <musa_runtime_api.h>

namespace infini::ops::moore_utils {

inline void CheckMusa(musaError_t err, const char* msg) {
  assert((err == musaSuccess) && msg);
}

inline musaStream_t GetMusaStream(void* stream) {
  return static_cast<musaStream_t>(stream ? stream : nullptr);
}

template <typename T>
inline void FreeDevice(T*& ptr) {
  if (ptr == nullptr) {
    return;
  }

  musaFree(reinterpret_cast<void*>(ptr));
  ptr = nullptr;
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

}  // namespace infini::ops::moore_utils

#endif
