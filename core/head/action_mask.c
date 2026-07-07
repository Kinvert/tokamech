#include "core/head/action_mask.h"

int tkm_action_mask_argmax(
    const float* values,
    const uint8_t* valid_mask,
    uint32_t count,
    uint16_t* out_index
) {
    uint8_t found = 0;
    uint16_t best_index = 0;
    float best_value = 0.0f;

    if (!values || !valid_mask || !out_index || count == 0) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < count; i++) {
        if (!valid_mask[i]) {
            continue;
        }
        if (!found || values[i] > best_value) {
            found = 1;
            best_index = (uint16_t)i;
            best_value = values[i];
        }
    }

    if (!found) {
        return TKM_ERR;
    }

    *out_index = best_index;
    return TKM_OK;
}
