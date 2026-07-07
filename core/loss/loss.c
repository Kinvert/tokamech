#include "core/loss/loss.h"

#include <math.h>

#define TKM_LOSS_EPSILON 0.0000001f

static float tkm_loss_abs(float value) {
    return value < 0.0f ? -value : value;
}

static int tkm_loss_valid_probability(float probability) {
    return probability >= 0.0f && probability <= 1.0f;
}

static float tkm_loss_clamp_probability(float probability) {
    if (probability < TKM_LOSS_EPSILON) {
        return TKM_LOSS_EPSILON;
    }
    if (probability > 1.0f - TKM_LOSS_EPSILON) {
        return 1.0f - TKM_LOSS_EPSILON;
    }
    return probability;
}

int tkm_loss_cross_entropy(const float* probabilities, uint32_t count, uint32_t target, float* out_loss) {
    if (!probabilities || !out_loss || count == 0 || target >= count) {
        return TKM_ERR;
    }
    if (!tkm_loss_valid_probability(probabilities[target])) {
        return TKM_ERR;
    }

    *out_loss = -logf(tkm_loss_clamp_probability(probabilities[target]));
    return TKM_OK;
}

int tkm_loss_mse(const float* predicted, const float* target, uint32_t count, float* out_loss) {
    float total = 0.0f;

    if (!predicted || !target || !out_loss || count == 0) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < count; i++) {
        float error = predicted[i] - target[i];
        total += error * error;
    }

    *out_loss = total / (float)count;
    return TKM_OK;
}

int tkm_loss_huber(const float* predicted, const float* target, uint32_t count, float delta, float* out_loss) {
    float total = 0.0f;

    if (!predicted || !target || !out_loss || count == 0 || delta <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < count; i++) {
        float error = predicted[i] - target[i];
        float abs_error = tkm_loss_abs(error);
        if (abs_error <= delta) {
            total += 0.5f * error * error;
        } else {
            total += delta * (abs_error - 0.5f * delta);
        }
    }

    *out_loss = total / (float)count;
    return TKM_OK;
}

int tkm_loss_binary_cross_entropy(float probability, uint8_t target, float* out_loss) {
    float clipped;

    if (!out_loss || target > 1 || !tkm_loss_valid_probability(probability)) {
        return TKM_ERR;
    }

    clipped = tkm_loss_clamp_probability(probability);
    *out_loss = target ? -logf(clipped) : -logf(1.0f - clipped);
    return TKM_OK;
}

int tkm_loss_weighted_sum(const float* losses, const float* weights, uint32_t count, float* out_loss) {
    float total = 0.0f;

    if (!losses || !weights || !out_loss || count == 0) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < count; i++) {
        total += losses[i] * weights[i];
    }

    *out_loss = total;
    return TKM_OK;
}

int tkm_loss_action_smoothness(const float* actions, uint32_t step_count, uint32_t action_dim, float* out_loss) {
    float total = 0.0f;
    uint32_t transition_count;

    if (!actions || !out_loss || step_count < 2 || action_dim == 0) {
        return TKM_ERR;
    }

    transition_count = step_count - 1;
    for (uint32_t step = 1; step < step_count; step++) {
        for (uint32_t dim = 0; dim < action_dim; dim++) {
            float previous = actions[(step - 1) * action_dim + dim];
            float current = actions[step * action_dim + dim];
            float diff = current - previous;
            total += diff * diff;
        }
    }

    *out_loss = total / (float)transition_count;
    return TKM_OK;
}
