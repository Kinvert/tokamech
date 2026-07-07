#include "core/train/action_scorer_trainer.h"

int tkm_action_scorer_train_configured(
    TkmActionScorer* scorer,
    TkmTrainLossKind loss,
    const float* state_features,
    const float* action_features,
    const uint8_t* valid_mask,
    uint32_t action_count,
    uint16_t target_action,
    float margin,
    float learning_rate
) {
    if (!scorer ||
        !state_features ||
        !action_features ||
        !valid_mask ||
        action_count == 0 ||
        action_count > TKM_ACTION_SCORER_MAX_ACTIONS ||
        target_action >= action_count ||
        !valid_mask[target_action] ||
        learning_rate <= 0.0f) {
        return TKM_ERR;
    }

    if (loss == TKM_TRAIN_LOSS_MASKED_CROSS_ENTROPY) {
        return tkm_action_scorer_train_masked_cross_entropy(
            scorer,
            state_features,
            action_features,
            valid_mask,
            action_count,
            target_action,
            learning_rate
        );
    }

    if (loss == TKM_TRAIN_LOSS_MARGIN) {
        if (margin <= 0.0f) {
            return TKM_ERR;
        }
        for (uint32_t action = 0; action < action_count; action++) {
            if (!valid_mask[action] || action == target_action) {
                continue;
            }
            if (tkm_action_scorer_train_margin(
                    scorer,
                    state_features,
                    action_features + (uint32_t)target_action * scorer->action_dim,
                    action_features + action * scorer->action_dim,
                    margin,
                    learning_rate
                ) != TKM_OK) {
                return TKM_ERR;
            }
        }
        return TKM_OK;
    }

    return TKM_ERR;
}
