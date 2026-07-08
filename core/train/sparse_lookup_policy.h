#ifndef TKM_SPARSE_LOOKUP_POLICY_H
#define TKM_SPARSE_LOOKUP_POLICY_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/dataset/trajectory.h"

#define TKM_SPARSE_LOOKUP_MAX_ITEMS 512u
#define TKM_SPARSE_LOOKUP_MAX_ACTIONS 16u

typedef struct {
    int32_t tokens[TKM_SPARSE_LOOKUP_MAX_ITEMS];
    uint8_t action_for_item[TKM_SPARSE_LOOKUP_MAX_ITEMS];
    uint32_t item_count;
    uint8_t num_actions;
    uint8_t default_action;
} TkmSparseLookupPolicy;

int tkm_sparse_lookup_policy_train(
    TkmSparseLookupPolicy* policy,
    const TkmTrajectory* trajectory,
    uint8_t num_actions,
    uint8_t default_action
);
int tkm_sparse_lookup_policy_train_return_weighted(
    TkmSparseLookupPolicy* policy,
    const TkmTrajectory* trajectory,
    uint8_t num_actions,
    uint8_t default_action,
    float discount
);
uint8_t tkm_sparse_lookup_policy_predict(const TkmSparseLookupPolicy* policy, int32_t token);

#endif
