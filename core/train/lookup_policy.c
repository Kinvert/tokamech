#include "core/train/lookup_policy.h"

#include <stdlib.h>

int tkm_lookup_policy_train(
    TkmLookupPolicy* policy,
    const TkmIntBins* tokenizer,
    const TkmTrajectory* trajectory,
    uint8_t num_actions
) {
    uint16_t* counts;
    int status = TKM_OK;

    if (tokenizer->vocab_size > TKM_LOOKUP_MAX_TOKENS || num_actions > TKM_LOOKUP_MAX_ACTIONS || num_actions == 0) {
        return TKM_ERR;
    }

    counts = (uint16_t*)calloc((size_t)tokenizer->vocab_size * (size_t)num_actions, sizeof(*counts));
    if (!counts) {
        return TKM_ERR;
    }

    policy->vocab_size = tokenizer->vocab_size;
    policy->num_actions = num_actions;
    for (uint16_t token = 0; token < TKM_LOOKUP_MAX_TOKENS; token++) {
        policy->action_for_token[token] = 0;
        policy->has_action[token] = 0;
    }

    for (uint32_t i = 0; i < trajectory->len; i++) {
        uint16_t token = tkm_int_bins_encode(tokenizer, trajectory->obs_i32[i]);
        uint8_t action = trajectory->actions[i];

        if (token == TKM_INVALID_TOKEN || token >= tokenizer->vocab_size || action >= num_actions) {
            status = TKM_ERR;
            break;
        }

        counts[(size_t)token * (size_t)num_actions + (size_t)action]++;
    }

    if (status == TKM_OK) {
        for (uint16_t token = 0; token < tokenizer->vocab_size; token++) {
            uint16_t best_count = 0;
            uint8_t best_action = 0;

            for (uint8_t action = 0; action < num_actions; action++) {
                uint16_t count = counts[(size_t)token * (size_t)num_actions + (size_t)action];
                if (count > best_count) {
                    best_count = count;
                    best_action = action;
                }
            }

            if (best_count > 0) {
                policy->action_for_token[token] = best_action;
                policy->has_action[token] = 1;
            }
        }
    }

    free(counts);
    return status;
}

uint8_t tkm_lookup_policy_predict(const TkmLookupPolicy* policy, uint16_t token) {
    if (token >= policy->vocab_size || token >= TKM_LOOKUP_MAX_TOKENS || !policy->has_action[token]) {
        return 0;
    }

    return policy->action_for_token[token];
}
