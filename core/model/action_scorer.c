#include "core/model/action_scorer.h"

#include <float.h>
#include <math.h>

#include "core/head/action_mask.h"

static uint32_t tkm_action_scorer_input_dim(const TkmActionScorer* scorer) {
    return scorer->state_dim + scorer->action_dim;
}

static uint32_t tkm_action_scorer_w0_offset(const TkmActionScorer* scorer, uint32_t input_index, uint32_t hidden_index) {
    return input_index * scorer->mlp.hidden_dim + hidden_index;
}

static uint32_t tkm_action_scorer_w1_offset(uint32_t hidden_index) {
    return hidden_index;
}

static float tkm_action_scorer_relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

static float tkm_action_scorer_input_value(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    uint32_t input_index
) {
    if (input_index < scorer->state_dim) {
        return state_features[input_index];
    }
    return action_features[input_index - scorer->state_dim];
}

static int tkm_action_scorer_ready(const TkmActionScorer* scorer) {
    return scorer &&
        scorer->state_dim > 0 &&
        scorer->action_dim > 0 &&
        scorer->mlp.input_dim == tkm_action_scorer_input_dim(scorer) &&
        scorer->mlp.output_dim == 1u;
}

static int tkm_action_scorer_forward_cache(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    float* out_pre_hidden,
    float* out_hidden,
    float* out_score
) {
    if (!tkm_action_scorer_ready(scorer) ||
        !state_features ||
        !action_features ||
        !out_pre_hidden ||
        !out_hidden ||
        !out_score) {
        return TKM_ERR;
    }

    for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
        float value = scorer->mlp.b0[h];
        for (uint32_t i = 0; i < scorer->mlp.input_dim; i++) {
            value += tkm_action_scorer_input_value(scorer, state_features, action_features, i) *
                scorer->mlp.w0[tkm_action_scorer_w0_offset(scorer, i, h)];
        }
        out_pre_hidden[h] = value;
        out_hidden[h] = tkm_action_scorer_relu(value);
    }

    *out_score = scorer->mlp.b1[0];
    for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
        *out_score += out_hidden[h] * scorer->mlp.w1[tkm_action_scorer_w1_offset(h)];
    }

    return TKM_OK;
}

int tkm_action_scorer_init(
    TkmActionScorer* scorer,
    uint32_t state_dim,
    uint32_t action_dim,
    uint32_t hidden_dim,
    const float* w0,
    const float* b0,
    const float* w1,
    const float* b1
) {
    if (!scorer || state_dim == 0 || action_dim == 0) {
        return TKM_ERR;
    }
    if (state_dim + action_dim > TKM_MLP_WINDOW_MAX_INPUT) {
        return TKM_ERR;
    }

    scorer->state_dim = state_dim;
    scorer->action_dim = action_dim;
    return tkm_mlp_window_init(&scorer->mlp, state_dim + action_dim, hidden_dim, 1u, w0, b0, w1, b1);
}

int tkm_action_scorer_score(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    float* out_score
) {
    float pre_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];

    return tkm_action_scorer_forward_cache(
        scorer,
        state_features,
        action_features,
        pre_hidden,
        hidden,
        out_score
    );
}

int tkm_action_scorer_score_all(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    uint32_t action_count,
    float* out_scores,
    uint32_t max_scores
) {
    if (!tkm_action_scorer_ready(scorer) ||
        !state_features ||
        !action_features ||
        !out_scores ||
        action_count == 0 ||
        max_scores < action_count) {
        return TKM_ERR;
    }

    for (uint32_t action = 0; action < action_count; action++) {
        const float* action_row = action_features + action * scorer->action_dim;
        if (tkm_action_scorer_score(scorer, state_features, action_row, &out_scores[action]) != TKM_OK) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

int tkm_action_scorer_masked_argmax(
    const TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    const uint8_t* valid_mask,
    uint32_t action_count,
    uint16_t* out_action
) {
    float scores[TKM_ACTION_SCORER_MAX_ACTIONS];

    if (!valid_mask || !out_action || action_count == 0 || action_count > TKM_ACTION_SCORER_MAX_ACTIONS) {
        return TKM_ERR;
    }
    if (tkm_action_scorer_score_all(scorer, state_features, action_features, action_count, scores, action_count) != TKM_OK) {
        return TKM_ERR;
    }

    return tkm_action_mask_argmax(scores, valid_mask, action_count, out_action);
}

int tkm_action_scorer_train_margin(
    TkmActionScorer* scorer,
    const float* state_features,
    const float* expert_action_features,
    const float* other_action_features,
    float margin,
    float learning_rate
) {
    float expert_pre[TKM_MLP_WINDOW_MAX_HIDDEN];
    float expert_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float other_pre[TKM_MLP_WINDOW_MAX_HIDDEN];
    float other_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float old_w1[TKM_MLP_WINDOW_MAX_HIDDEN];
    float expert_score = 0.0f;
    float other_score = 0.0f;

    if (!tkm_action_scorer_ready(scorer) ||
        !state_features ||
        !expert_action_features ||
        !other_action_features ||
        margin <= 0.0f ||
        learning_rate <= 0.0f) {
        return TKM_ERR;
    }
    if (tkm_action_scorer_forward_cache(
            scorer,
            state_features,
            expert_action_features,
            expert_pre,
            expert_hidden,
            &expert_score
        ) != TKM_OK ||
        tkm_action_scorer_forward_cache(
            scorer,
            state_features,
            other_action_features,
            other_pre,
            other_hidden,
            &other_score
        ) != TKM_OK) {
        return TKM_ERR;
    }

    if (expert_score >= other_score + margin) {
        return TKM_OK;
    }

    for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
        old_w1[h] = scorer->mlp.w1[tkm_action_scorer_w1_offset(h)];
    }

    for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
        float expert_gate = expert_pre[h] > 0.0f ? 1.0f : 0.0f;
        float other_gate = other_pre[h] > 0.0f ? 1.0f : 0.0f;
        float expert_hidden_grad = old_w1[h] * expert_gate;
        float other_hidden_grad = old_w1[h] * other_gate;

        scorer->mlp.w1[tkm_action_scorer_w1_offset(h)] +=
            learning_rate * (expert_hidden[h] - other_hidden[h]);

        for (uint32_t i = 0; i < scorer->mlp.input_dim; i++) {
            float expert_input = tkm_action_scorer_input_value(scorer, state_features, expert_action_features, i);
            float other_input = tkm_action_scorer_input_value(scorer, state_features, other_action_features, i);
            scorer->mlp.w0[tkm_action_scorer_w0_offset(scorer, i, h)] +=
                learning_rate * (expert_hidden_grad * expert_input - other_hidden_grad * other_input);
        }

        scorer->mlp.b0[h] += learning_rate * (expert_hidden_grad - other_hidden_grad);
    }

    return TKM_OK;
}

int tkm_action_scorer_train_masked_cross_entropy(
    TkmActionScorer* scorer,
    const float* state_features,
    const float* action_features,
    const uint8_t* valid_mask,
    uint32_t action_count,
    uint16_t target_action,
    float learning_rate
) {
    float scores[TKM_ACTION_SCORER_MAX_ACTIONS];
    float exp_scores[TKM_ACTION_SCORER_MAX_ACTIONS];
    float grad_w0[TKM_MLP_WINDOW_MAX_INPUT * TKM_MLP_WINDOW_MAX_HIDDEN] = {0.0f};
    float grad_b0[TKM_MLP_WINDOW_MAX_HIDDEN] = {0.0f};
    float grad_w1[TKM_MLP_WINDOW_MAX_HIDDEN] = {0.0f};
    float old_w1[TKM_MLP_WINDOW_MAX_HIDDEN];
    float pre_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float grad_b1 = 0.0f;

    if (!tkm_action_scorer_ready(scorer) ||
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

    if (tkm_action_scorer_score_all(scorer, state_features, action_features, action_count, scores, action_count) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t action = 0; action < action_count; action++) {
        if (valid_mask[action] && scores[action] > max_score) {
            max_score = scores[action];
        }
    }

    for (uint32_t action = 0; action < action_count; action++) {
        if (valid_mask[action]) {
            exp_scores[action] = expf(scores[action] - max_score);
            sum_exp += exp_scores[action];
        } else {
            exp_scores[action] = 0.0f;
        }
    }
    if (sum_exp <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
        old_w1[h] = scorer->mlp.w1[tkm_action_scorer_w1_offset(h)];
    }

    for (uint32_t action = 0; action < action_count; action++) {
        const float* action_row = action_features + action * scorer->action_dim;
        float action_score = 0.0f;
        float probability;
        float grad_score;

        if (!valid_mask[action]) {
            continue;
        }

        if (tkm_action_scorer_forward_cache(
                scorer,
                state_features,
                action_row,
                pre_hidden,
                hidden,
                &action_score
            ) != TKM_OK) {
            return TKM_ERR;
        }

        probability = exp_scores[action] / sum_exp;
        grad_score = probability - (action == target_action ? 1.0f : 0.0f);
        grad_b1 += grad_score;

        for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
            float gate = pre_hidden[h] > 0.0f ? 1.0f : 0.0f;
            float hidden_grad = grad_score * old_w1[h] * gate;

            grad_w1[tkm_action_scorer_w1_offset(h)] += grad_score * hidden[h];
            grad_b0[h] += hidden_grad;

            for (uint32_t i = 0; i < scorer->mlp.input_dim; i++) {
                grad_w0[tkm_action_scorer_w0_offset(scorer, i, h)] +=
                    hidden_grad * tkm_action_scorer_input_value(scorer, state_features, action_row, i);
            }
        }
    }

    scorer->mlp.b1[0] -= learning_rate * grad_b1;
    for (uint32_t h = 0; h < scorer->mlp.hidden_dim; h++) {
        scorer->mlp.w1[tkm_action_scorer_w1_offset(h)] -= learning_rate * grad_w1[tkm_action_scorer_w1_offset(h)];
        scorer->mlp.b0[h] -= learning_rate * grad_b0[h];
        for (uint32_t i = 0; i < scorer->mlp.input_dim; i++) {
            scorer->mlp.w0[tkm_action_scorer_w0_offset(scorer, i, h)] -=
                learning_rate * grad_w0[tkm_action_scorer_w0_offset(scorer, i, h)];
        }
    }

    return TKM_OK;
}
