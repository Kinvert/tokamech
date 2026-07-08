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

static void tkm_lookup_policy_compute_discounted_returns(
    const TkmTrajectory* trajectory,
    float discount,
    float* returns
) {
    uint32_t start = 0u;

    while (start < trajectory->len) {
        uint32_t end = start;
        float running = 0.0f;

        while (end + 1u < trajectory->len && !trajectory->terminals[end]) {
            end++;
        }

        for (uint32_t cursor = end + 1u; cursor > start; cursor--) {
            uint32_t i = cursor - 1u;
            running = trajectory->rewards[i] + discount * running;
            returns[i] = running;
        }

        start = end + 1u;
    }
}

int tkm_lookup_policy_train_return_weighted(
    TkmLookupPolicy* policy,
    const TkmIntBins* tokenizer,
    const TkmTrajectory* trajectory,
    uint8_t num_actions,
    float discount
) {
    float* scores;
    uint16_t* counts;
    float returns[TKM_TRAJECTORY_MAX_STEPS] = {0.0f};
    int status = TKM_OK;

    if (!policy || !tokenizer || !trajectory ||
        tokenizer->vocab_size > TKM_LOOKUP_MAX_TOKENS ||
        num_actions > TKM_LOOKUP_MAX_ACTIONS ||
        num_actions == 0 ||
        discount < 0.0f ||
        discount > 1.0f) {
        return TKM_ERR;
    }

    scores = (float*)calloc((size_t)tokenizer->vocab_size * (size_t)num_actions, sizeof(*scores));
    counts = (uint16_t*)calloc((size_t)tokenizer->vocab_size * (size_t)num_actions, sizeof(*counts));
    if (!scores || !counts) {
        free(scores);
        free(counts);
        return TKM_ERR;
    }

    policy->vocab_size = tokenizer->vocab_size;
    policy->num_actions = num_actions;
    for (uint16_t token = 0; token < TKM_LOOKUP_MAX_TOKENS; token++) {
        policy->action_for_token[token] = 0;
        policy->has_action[token] = 0;
    }
    tkm_lookup_policy_compute_discounted_returns(trajectory, discount, returns);

    for (uint32_t i = 0; i < trajectory->len; i++) {
        uint16_t token = tkm_int_bins_encode(tokenizer, trajectory->obs_i32[i]);
        uint8_t action = trajectory->actions[i];
        size_t offset;

        if (token == TKM_INVALID_TOKEN || token >= tokenizer->vocab_size || action >= num_actions) {
            status = TKM_ERR;
            break;
        }

        offset = (size_t)token * (size_t)num_actions + (size_t)action;
        scores[offset] += returns[i];
        counts[offset]++;
    }

    if (status == TKM_OK) {
        for (uint16_t token = 0; token < tokenizer->vocab_size; token++) {
            float best_score = 0.0f;
            uint16_t best_count = 0;
            uint8_t best_action = 0;
            uint8_t best_seen = 0;

            for (uint8_t action = 0; action < num_actions; action++) {
                size_t offset = (size_t)token * (size_t)num_actions + (size_t)action;
                if (counts[offset] == 0) {
                    continue;
                }
                if (!best_seen ||
                    scores[offset] > best_score ||
                    (scores[offset] == best_score && counts[offset] > best_count)) {
                    best_seen = 1;
                    best_score = scores[offset];
                    best_count = counts[offset];
                    best_action = action;
                }
            }

            if (best_seen) {
                policy->action_for_token[token] = best_action;
                policy->has_action[token] = 1;
            }
        }
    }

    free(scores);
    free(counts);
    return status;
}

uint8_t tkm_lookup_policy_predict(const TkmLookupPolicy* policy, uint16_t token) {
    if (token >= policy->vocab_size || token >= TKM_LOOKUP_MAX_TOKENS || !policy->has_action[token]) {
        return 0;
    }

    return policy->action_for_token[token];
}
