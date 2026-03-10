#ifndef INFINI_OPS_MOORE_ADD_LAUNCH_H_
#define INFINI_OPS_MOORE_ADD_LAUNCH_H_

#include <cstddef>

#include <musa_runtime_api.h>

namespace infini::ops::add::moore {

musaError_t LaunchAdd(const void* input, const void* other, void* out,
                      const size_t* out_shape, const size_t* input_shape,
                      const size_t* other_shape, const ptrdiff_t* out_strides,
                      const ptrdiff_t* input_strides,
                      const ptrdiff_t* other_strides, size_t output_size,
                      size_t ndim, bool out_contiguous, bool input_contiguous,
                      bool other_contiguous, int dtype, musaStream_t stream);

}  // namespace infini::ops::add::moore

#endif  // INFINI_OPS_MOORE_ADD_LAUNCH_H_
