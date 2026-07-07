#include "core/train/sparse_lookup_policy.h"

static int tkm_sparse_lookup_find(const TkmSparseLookupPolicy* policy, int32_t token, uint32_t* out_index) {
    if (!policy || !out_index) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < policy->item_count; i++) {
        if (policy->tokens[i] == token) {
            *out_index = i;
            return TKM_OK;
        }
    }

    return TKM_ERR;
}

int tkm_sparse_lookup_policy_train(
    TkmSparseLookupPolicy* policy,
    const TkmTrajectory* trajectory,
    uint8_t num_actions,
    uint8_t default_action
) {
    uint16_t counts[TKM_SPARSE_LOOKUP_MAX_ITEMS][TKM_SPARSE_LOOKUP_MAX_ACTIONS] = {{0}};

    if (!policy || !trajectory || num_actions == 0 || num_actions > TKM_SPARSE_LOOKUP_MAX_ACTIONS ||
        default_action >= num_actions) {
        return TKM_ERR;
    }

    policy->item_count = 0;
    policy->num_actions = num_actions;
    policy->default_action = default_action;

    for (uint32_t i = 0; i < trajectory->len; i++) {
        int32_t token = trajectory->obs_i32[i];
        uint8_t action = trajectory->actions[i];
        uint32_t index = 0;

        if (action >= num_actions) {
            return TKM_ERR;
        }

        if (tkm_sparse_lookup_find(policy, token, &index) != TKM_OK) {
            if (policy->item_count >= TKM_SPARSE_LOOKUP_MAX_ITEMS) {
                return TKM_ERR;
            }
            index = policy->item_count++;
            policy->tokens[index] = token;
            policy->action_for_item[index] = default_action;
        }

        counts[index][action]++;
    }

    for (uint32_t i = 0; i < policy->item_count; i++) {
        uint16_t best_count = 0;
        uint8_t best_action = default_action;

        for (uint8_t action = 0; action < num_actions; action++) {
            if (counts[i][action] > best_count) {
                best_count = counts[i][action];
                best_action = action;
            }
        }

        policy->action_for_item[i] = best_action;
    }

    return TKM_OK;
}

uint8_t tkm_sparse_lookup_policy_predict(const TkmSparseLookupPolicy* policy, int32_t token) {
    uint32_t index = 0;

    if (!policy) {
        return 0;
    }

    if (tkm_sparse_lookup_find(policy, token, &index) == TKM_OK) {
        return policy->action_for_item[index];
    }

    return policy->default_action;
}
