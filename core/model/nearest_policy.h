#ifndef TKM_NEAREST_POLICY_H
#define TKM_NEAREST_POLICY_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_NEAREST_POLICY_MAX_ITEMS 512u
#define TKM_NEAREST_POLICY_MAX_DIM 256u
#define TKM_NEAREST_POLICY_MAX_ACTIONS 16u

typedef struct {
    uint32_t dim;
    uint32_t item_count;
    uint8_t num_actions;
    uint8_t default_action;
    float features[TKM_NEAREST_POLICY_MAX_ITEMS * TKM_NEAREST_POLICY_MAX_DIM];
    uint8_t actions[TKM_NEAREST_POLICY_MAX_ITEMS];
} TkmNearestPolicy;

int tkm_nearest_policy_init(
    TkmNearestPolicy* policy,
    uint32_t dim,
    uint8_t num_actions,
    uint8_t default_action
);
int tkm_nearest_policy_add(TkmNearestPolicy* policy, const float* features, uint8_t action);
int tkm_nearest_policy_predict(const TkmNearestPolicy* policy, const float* features, uint8_t* out_action);

#endif
