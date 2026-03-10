#ifndef INFINI_OPS_MOORE_RMS_NORM_LAUNCH_H_
#define INFINI_OPS_MOORE_RMS_NORM_LAUNCH_H_

#include <cstddef>

#include <musa_runtime_api.h>

namespace infini::ops::rms_norm::moore {

musaError_t LaunchRmsNorm(void* out, const void* input, const void* weight,
                          ptrdiff_t stride_out_batch,
                          ptrdiff_t stride_out_nhead,
                          ptrdiff_t stride_input_batch,
                          ptrdiff_t stride_input_nhead, size_t batch_size,
                          size_t nhead, size_t dim, float epsilon, int dtype,
                          musaStream_t stream);

}  // namespace infini::ops::rms_norm::moore

#endif  // INFINI_OPS_MOORE_RMS_NORM_LAUNCH_H_
