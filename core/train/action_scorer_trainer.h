#ifndef TKM_ACTION_SCORER_TRAINER_H
#define TKM_ACTION_SCORER_TRAINER_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/config/train_config.h"
#include "core/model/action_scorer.h"

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
);

#endif
