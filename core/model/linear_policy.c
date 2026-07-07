#include "core/model/linear_policy.h"

static uint32_t tkm_linear_policy_offset(const TkmLinearPolicy* policy, uint8_t action, uint32_t input_index) {
    return (uint32_t)action * policy->input_dim + input_index;
}

int tkm_linear_policy_init(TkmLinearPolicy* policy, uint32_t input_dim, uint8_t num_actions) {
    if (!policy || input_dim == 0 || input_dim > TKM_LINEAR_POLICY_MAX_INPUT ||
        num_actions == 0 || num_actions > TKM_LINEAR_POLICY_MAX_ACTIONS) {
        return TKM_ERR;
    }

    policy->input_dim = input_dim;
    policy->num_actions = num_actions;

    for (uint32_t i = 0; i < TKM_LINEAR_POLICY_MAX_INPUT * TKM_LINEAR_POLICY_MAX_ACTIONS; i++) {
        policy->weights[i] = 0.0f;
    }
    for (uint8_t action = 0; action < TKM_LINEAR_POLICY_MAX_ACTIONS; action++) {
        policy->bias[action] = 0.0f;
    }

    return TKM_OK;
}

int tkm_linear_policy_logits(
    const TkmLinearPolicy* policy,
    const float* input,
    float* out_logits,
    uint32_t out_count
) {
    if (!policy || !input || !out_logits || policy->input_dim == 0 || policy->num_actions == 0 ||
        out_count < policy->num_actions) {
        return TKM_ERR;
    }

    for (uint8_t action = 0; action < policy->num_actions; action++) {
        float value = policy->bias[action];
        for (uint32_t i = 0; i < policy->input_dim; i++) {
            value += input[i] * policy->weights[tkm_linear_policy_offset(policy, action, i)];
        }
        out_logits[action] = value;
    }

    return TKM_OK;
}

int tkm_linear_policy_predict(const TkmLinearPolicy* policy, const float* input, uint8_t* out_action) {
    float logits[TKM_LINEAR_POLICY_MAX_ACTIONS];
    uint8_t best_action = 0;

    if (!policy || !input || !out_action) {
        return TKM_ERR;
    }
    if (tkm_linear_policy_logits(policy, input, logits, TKM_LINEAR_POLICY_MAX_ACTIONS) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint8_t action = 1; action < policy->num_actions; action++) {
        if (logits[action] > logits[best_action]) {
            best_action = action;
        }
    }

    *out_action = best_action;
    return TKM_OK;
}

int tkm_linear_policy_train_perceptron(
    TkmLinearPolicy* policy,
    const float* input,
    uint8_t target_action,
    float learning_rate
) {
    uint8_t predicted_action = 0;

    if (!policy || !input || target_action >= policy->num_actions || learning_rate <= 0.0f) {
        return TKM_ERR;
    }
    if (tkm_linear_policy_predict(policy, input, &predicted_action) != TKM_OK) {
        return TKM_ERR;
    }
    if (predicted_action == target_action) {
        return TKM_OK;
    }

    for (uint32_t i = 0; i < policy->input_dim; i++) {
        policy->weights[tkm_linear_policy_offset(policy, target_action, i)] += learning_rate * input[i];
        policy->weights[tkm_linear_policy_offset(policy, predicted_action, i)] -= learning_rate * input[i];
    }
    policy->bias[target_action] += learning_rate;
    policy->bias[predicted_action] -= learning_rate;

    return TKM_OK;
}
