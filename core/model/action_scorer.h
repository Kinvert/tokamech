#ifndef TKM_ACTION_SCORER_H
#define TKM_ACTION_SCORER_H

#include <stdint.h>

#include "core/common/status.h"
#include "core/model/mlp_window.h"

#define TKM_ACTION_SCORER_MAX_ACTIONS 256u

typedef struct {
    uint32_t state_dim;
    uint32_t action_dim;
    TkmMlpWindow mlp;
} TkmActionScorer;

int tkm_action_scorer_init(
    TkmActionScorer* scorer,
    uint32_t state_dim,
    uint32_t action_dim,
    uint32_t hidden_dim,
    const float* w0,
    const float* b0,
    const float* w1,
    const float* b1
);
int tkm_action_scorer_score(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    float* out_score
);
int tkm_action_scorer_score_all(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    uint32_t action_count,
    float* out_scores,
    uint32_t max_scores
);
int tkm_action_scorer_masked_argmax(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    const uint8_t* valid_mask,
    uint32_t action_count,
    uint16_t* out_action
);
int tkm_action_scorer_train_margin(
    TkmActionScorer* scorer,
    const float* state_features,
    const float* expert_action_features,
    const float* other_action_features,
    float margin,
    float learning_rate
);
int tkm_action_scorer_train_masked_cross_entropy(
    TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    const uint8_t* valid_mask,
    uint32_t action_count,
    uint16_t target_action,
    float learning_rate
);

#endif
