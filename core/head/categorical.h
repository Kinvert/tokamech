#ifndef TKM_CATEGORICAL_HEAD_H
#define TKM_CATEGORICAL_HEAD_H

#include <stdint.h>

#include "core/common/status.h"

int tkm_categorical_softmax(const float* logits, uint32_t count, float* out_probs);
int tkm_categorical_argmax(const float* values, uint32_t count, uint16_t* out_index);

#endif
