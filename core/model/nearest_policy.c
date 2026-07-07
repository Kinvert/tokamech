#include "core/model/nearest_policy.h"

static uint32_t tkm_nearest_policy_offset(const TkmNearestPolicy* policy, uint32_t item, uint32_t dim) {
    return item * policy->dim + dim;
}

int tkm_nearest_policy_init(
    TkmNearestPolicy* policy,
    uint32_t dim,
    uint8_t num_actions,
    uint8_t default_action
) {
    if (!policy || dim == 0 || dim > TKM_NEAREST_POLICY_MAX_DIM ||
        num_actions == 0 || num_actions > TKM_NEAREST_POLICY_MAX_ACTIONS ||
        default_action >= num_actions) {
        return TKM_ERR;
    }

    policy->dim = dim;
    policy->item_count = 0;
    policy->num_actions = num_actions;
    policy->default_action = default_action;
    return TKM_OK;
}

int tkm_nearest_policy_add(TkmNearestPolicy* policy, const float* features, uint8_t action) {
    uint32_t item;

    if (!policy || !features || policy->dim == 0 || action >= policy->num_actions ||
        policy->item_count >= TKM_NEAREST_POLICY_MAX_ITEMS) {
        return TKM_ERR;
    }

    item = policy->item_count++;
    for (uint32_t i = 0; i < policy->dim; i++) {
        policy->features[tkm_nearest_policy_offset(policy, item, i)] = features[i];
    }
    policy->actions[item] = action;

    return TKM_OK;
}

int tkm_nearest_policy_predict(const TkmNearestPolicy* policy, const float* features, uint8_t* out_action) {
    float best_distance = 0.0f;
    uint32_t best_item = 0;

    if (!policy || !features || !out_action || policy->dim == 0) {
        return TKM_ERR;
    }

    if (policy->item_count == 0) {
        *out_action = policy->default_action;
        return TKM_OK;
    }

    for (uint32_t item = 0; item < policy->item_count; item++) {
        float distance = 0.0f;

        for (uint32_t i = 0; i < policy->dim; i++) {
            float diff = features[i] - policy->features[tkm_nearest_policy_offset(policy, item, i)];
            distance += diff * diff;
        }

        if (item == 0 || distance < best_distance) {
            best_distance = distance;
            best_item = item;
        }
    }

    *out_action = policy->actions[best_item];
    return TKM_OK;
}
