#ifndef TKM_LINEAR_POLICY_H
#define TKM_LINEAR_POLICY_H

#include <stdint.h>

#include "core/common/status.h"

#define TKM_LINEAR_POLICY_MAX_INPUT 256u
#define TKM_LINEAR_POLICY_MAX_ACTIONS 16u

typedef struct {
    uint32_t input_dim;
    uint8_t num_actions;
    float weights[TKM_LINEAR_POLICY_MAX_INPUT * TKM_LINEAR_POLICY_MAX_ACTIONS];
    float bias[TKM_LINEAR_POLICY_MAX_ACTIONS];
} TkmLinearPolicy;

int tkm_linear_policy_init(TkmLinearPolicy* policy, uint32_t input_dim, uint8_t num_actions);
int tkm_linear_policy_logits(
    const TkmLinearPolicy* policy,
    const float* input,
    float* out_logits,
    uint32_t out_count
);
int tkm_linear_policy_predict(const TkmLinearPolicy* policy, const float* input, uint8_t* out_action);
int tkm_linear_policy_train_perceptron(
    TkmLinearPolicy* policy,
    const float* input,
    uint8_t target_action,
    float learning_rate
);

#endif
