#ifndef INFINI_OPS_BASE_NEG_H_
#define INFINI_OPS_BASE_NEG_H_

#include "base/unary_elementwise.h"

namespace infini::ops {

INFINI_OPS_DECLARE_UNARY_ELEMENTWISE_OP(Neg, Neg, SignedAndFloatTypes)

}  // namespace infini::ops

#endif
