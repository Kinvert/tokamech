#include "core/train/lookup_policy.h"

int tkm_lookup_policy_train(
    TkmLookupPolicy* policy,
    const TkmIntBins* tokenizer,
    const TkmTrajectory* trajectory,
    uint8_t num_actions
) {
    if (tokenizer->vocab_size > TKM_LOOKUP_MAX_TOKENS || num_actions > TKM_LOOKUP_MAX_ACTIONS || num_actions == 0) {
        return TKM_ERR;
    }

    uint16_t counts[TKM_LOOKUP_MAX_TOKENS][TKM_LOOKUP_MAX_ACTIONS] = {{0}};

    policy->vocab_size = tokenizer->vocab_size;
    policy->num_actions = num_actions;
    for (uint16_t token = 0; token < TKM_LOOKUP_MAX_TOKENS; token++) {
        policy->action_for_token[token] = 0;
        policy->has_action[token] = 0;
    }

    for (uint32_t i = 0; i < trajectory->len; i++) {
        uint16_t token = tkm_int_bins_encode(tokenizer, trajectory->obs_i16[i]);
        uint8_t action = trajectory->actions[i];

        if (token == TKM_INVALID_TOKEN || token >= tokenizer->vocab_size || action >= num_actions) {
            return TKM_ERR;
        }

        counts[token][action]++;
    }

    for (uint16_t token = 0; token < tokenizer->vocab_size; token++) {
        uint16_t best_count = 0;
        uint8_t best_action = 0;

        for (uint8_t action = 0; action < num_actions; action++) {
            if (counts[token][action] > best_count) {
                best_count = counts[token][action];
                best_action = action;
            }
        }

        if (best_count > 0) {
            policy->action_for_token[token] = best_action;
            policy->has_action[token] = 1;
        }
    }

    return TKM_OK;
}

uint8_t tkm_lookup_policy_predict(const TkmLookupPolicy* policy, uint16_t token) {
    if (token >= policy->vocab_size || token >= TKM_LOOKUP_MAX_TOKENS || !policy->has_action[token]) {
        return 0;
    }

    return policy->action_for_token[token];
}
