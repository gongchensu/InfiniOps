#ifndef INFINI_OPS_MOORE_CAUSAL_SOFTMAX_LAUNCH_H_
#define INFINI_OPS_MOORE_CAUSAL_SOFTMAX_LAUNCH_H_

#include <cstddef>

#include <musa_runtime_api.h>

namespace infini::ops::causal_softmax::moore {

musaError_t LaunchCausalSoftmax(void* out, const void* input, size_t batch_size,
                                size_t seq_len, size_t total_seq_len,
                                ptrdiff_t stride_out_batch,
                                ptrdiff_t stride_out_row,
                                ptrdiff_t stride_input_batch,
                                ptrdiff_t stride_input_row, int dtype,
                                musaStream_t stream);

}  // namespace infini::ops::causal_softmax::moore

#endif  // INFINI_OPS_MOORE_CAUSAL_SOFTMAX_LAUNCH_H_
