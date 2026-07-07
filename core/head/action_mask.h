#ifndef TKM_ACTION_MASK_H
#define TKM_ACTION_MASK_H

#include <stdint.h>

#include "core/common/status.h"

int tkm_action_mask_argmax(
    const float* values,
    const uint8_t* valid_mask,
    uint32_t count,
    uint16_t* out_index
);

#endif
