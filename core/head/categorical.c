#include "core/head/categorical.h"

#include <math.h>

int tkm_categorical_softmax(const float* logits, uint32_t count, float* out_probs) {
    float max_logit;
    float sum = 0.0f;

    if (!logits || !out_probs || count == 0) {
        return TKM_ERR;
    }

    max_logit = logits[0];
    for (uint32_t i = 1; i < count; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    for (uint32_t i = 0; i < count; i++) {
        out_probs[i] = expf(logits[i] - max_logit);
        sum += out_probs[i];
    }

    if (sum <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < count; i++) {
        out_probs[i] /= sum;
    }

    return TKM_OK;
}

int tkm_categorical_argmax(const float* values, uint32_t count, uint16_t* out_index) {
    uint16_t best_index = 0;
    float best_value;

    if (!values || !out_index || count == 0) {
        return TKM_ERR;
    }

    best_value = values[0];
    for (uint32_t i = 1; i < count; i++) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_index = (uint16_t)i;
        }
    }

    *out_index = best_index;
    return TKM_OK;
}
