#include "projects/breakout/pufferlib_policy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/sequence/layout.h"

static int breakout_copy_name(char* dst, uint32_t dst_count, const char* src) {
    uint32_t i = 0u;

    if (!dst || !src || dst_count == 0u) {
        return TKM_ERR;
    }
    while (src[i] != '\0') {
        if (i + 1u >= dst_count) {
            return TKM_ERR;
        }
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
    return TKM_OK;
}

static const char* breakout_token_model_name(BreakoutPufferlibTokenModelKind kind) {
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        return "mlp_window";
    }
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) {
        return "token_mlp_window";
    }
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM) {
        return "token_backoff_ngram";
    }
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM) {
        return "token_ngram";
    }
    return "token_ngram";
}

static const char* breakout_token_head_name(BreakoutPufferlibTokenHeadKind kind) {
    (void)kind;
    return "categorical";
}

static const char* breakout_token_layout_name(BreakoutPufferlibTokenModelKind kind) {
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        return "obs_action_interleaved";
    }
    return "autoregressive_obs_action";
}

static const char* breakout_token_observation_tokenizer_name(BreakoutPufferlibTokenModelKind kind) {
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        return "selected_continuous";
    }
    return "selected_quantized";
}

static const char* breakout_token_selection_name(BreakoutPufferlibTokenObservationSelectionKind kind) {
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS) {
        return "first_dims";
    }
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL) {
        return "manual";
    }
    return "first_dims";
}

static const char* breakout_token_input_feature_name(BreakoutPufferlibTokenInputFeatureKind kind) {
    if (kind == BREAKOUT_PUFFERLIB_TOKEN_INPUT_VALUE_WINDOW) {
        return "value_window";
    }
    return "token_stream";
}

static int breakout_token_model_uses_ngram(BreakoutPufferlibTokenModelKind kind) {
    return kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM ||
        kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM;
}

static int breakout_token_model_predicts_next_token(BreakoutPufferlibTokenModelKind kind) {
    return breakout_token_model_uses_ngram(kind) ||
        kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW;
}

static int breakout_token_model_uses_token_stream(BreakoutPufferlibTokenModelKind kind) {
    return breakout_token_model_predicts_next_token(kind);
}

static int breakout_token_model_predicts_action(BreakoutPufferlibTokenModelKind kind) {
    return kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW;
}

static int breakout_token_model_uses_value_window(BreakoutPufferlibTokenModelKind kind) {
    return breakout_token_model_predicts_action(kind);
}

static int breakout_token_model_kind_from_string(
    const char* text,
    BreakoutPufferlibTokenModelKind* out
) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (strcmp(text, "mlp_window") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW;
        return TKM_OK;
    }
    if (strcmp(text, "token_ngram") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM;
        return TKM_OK;
    }
    if (strcmp(text, "token_backoff_ngram") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM;
        return TKM_OK;
    }
    if (strcmp(text, "token_mlp_window") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW;
        return TKM_OK;
    }
    return TKM_ERR;
}

static int breakout_token_head_kind_from_string(
    const char* text,
    BreakoutPufferlibTokenHeadKind* out
) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (strcmp(text, "categorical") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL;
        return TKM_OK;
    }
    return TKM_ERR;
}

static int breakout_token_selection_kind_from_string(
    const char* text,
    BreakoutPufferlibTokenObservationSelectionKind* out
) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (strcmp(text, "first_dims") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS;
        return TKM_OK;
    }
    if (strcmp(text, "manual") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL;
        return TKM_OK;
    }
    return TKM_ERR;
}

static int breakout_token_input_feature_kind_from_string(
    const char* text,
    BreakoutPufferlibTokenInputFeatureKind* out
) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (strcmp(text, "token_stream") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_INPUT_TOKEN_STREAM;
        return TKM_OK;
    }
    if (strcmp(text, "value_window") == 0) {
        *out = BREAKOUT_PUFFERLIB_TOKEN_INPUT_VALUE_WINDOW;
        return TKM_OK;
    }
    return TKM_ERR;
}

int breakout_pufferlib_render_visualization_kind_from_string(
    const char* text,
    BreakoutPufferlibRenderVisualizationKind* out
) {
    if (!text || !out) {
        return TKM_ERR;
    }
    if (strcmp(text, "none") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_NONE;
        return TKM_OK;
    }
    if (strcmp(text, "topk") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_TOPK;
        return TKM_OK;
    }
    if (strcmp(text, "token_stream") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_TOKEN_STREAM;
        return TKM_OK;
    }
    if (strcmp(text, "context_grid") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_CONTEXT_GRID;
        return TKM_OK;
    }
    if (strcmp(text, "quantization") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_QUANTIZATION;
        return TKM_OK;
    }
    if (strcmp(text, "pipeline") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_PIPELINE;
        return TKM_OK;
    }
    if (strcmp(text, "transformer") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_TRANSFORMER;
        return TKM_OK;
    }
    if (strcmp(text, "continuous_mlp") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_CONTINUOUS_MLP;
        return TKM_OK;
    }
    if (strcmp(text, "all") == 0) {
        *out = BREAKOUT_PUFFERLIB_RENDER_VIS_ALL;
        return TKM_OK;
    }
    return TKM_ERR;
}

static int breakout_token_parse_manual_selected_dims(
    const char* text,
    uint32_t expected_count,
    uint32_t* out_dims
) {
    const char* cursor;

    if (!text || !out_dims || expected_count == 0u ||
        expected_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }
    cursor = text;
    for (uint32_t i = 0u; i < expected_count; i++) {
        char* end = NULL;
        unsigned long value;

        while (*cursor == ' ' || *cursor == '\t') {
            cursor++;
        }
        value = strtoul(cursor, &end, 10);
        if (end == cursor || value >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
            return TKM_ERR;
        }
        out_dims[i] = (uint32_t)value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') {
            cursor++;
        }
        if (i + 1u < expected_count) {
            if (*cursor != ',') {
                return TKM_ERR;
            }
            cursor++;
        }
    }
    while (*cursor == ' ' || *cursor == '\t') {
        cursor++;
    }
    return *cursor == '\0' ? TKM_OK : TKM_ERR;
}

static int breakout_dataset_ready(const TkmFloatTransitionDataset* dataset) {
    return dataset &&
        dataset->row_count > 0 &&
        dataset->obs_dim == TKM_PUFFERLIB_BREAKOUT_OBS_DIM &&
        dataset->action_count == TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
}

static int breakout_transition_row_ready(const TkmFloatTransition* row) {
    return row &&
        row->obs_dim == TKM_PUFFERLIB_BREAKOUT_OBS_DIM &&
        row->action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
}

static int breakout_is_heldout(uint32_t row_index, uint32_t heldout_stride) {
    return heldout_stride > 0u && row_index % heldout_stride == 0u;
}

static uint32_t breakout_token_hash_mix(uint32_t hash, uint32_t value) {
    hash ^= value + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
    return hash;
}

static uint8_t breakout_token_quantize(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t dim,
    float value
) {
    float min_value = policy->obs_min[dim];
    float max_value = policy->obs_max[dim];
    float span = max_value - min_value;
    float scaled;
    int32_t token;

    if (span <= 0.000001f || policy->bin_count <= 1u) {
        return 0u;
    }
    if (value < min_value) {
        value = min_value;
    }
    if (value > max_value) {
        value = max_value;
    }
    scaled = (value - min_value) / span;
    token = (int32_t)(scaled * (float)policy->bin_count);
    if (token < 0) {
        token = 0;
    }
    if (token >= (int32_t)policy->bin_count) {
        token = (int32_t)policy->bin_count - 1;
    }
    return (uint8_t)token;
}

static uint32_t breakout_token_mlp_input_dim(const BreakoutPufferlibTokenPolicy* policy) {
    if (!policy) {
        return 0u;
    }
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) {
        return policy->history * 2u;
    }
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        return (policy->history + 1u) * policy->selected_dim_count;
    }
    return 0u;
}

static uint32_t breakout_token_mlp_output_dim(const BreakoutPufferlibTokenPolicy* policy) {
    uint32_t obs_vocab;

    if (!policy) {
        return 0u;
    }
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        return TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
    }
    obs_vocab = policy->selected_dim_count * policy->bin_count;
    return TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT + obs_vocab;
}

static int breakout_token_make_selected_tokens(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    uint8_t* out_tokens
) {
    if (!policy || !obs || !out_tokens ||
        policy->selected_dim_count == 0u ||
        policy->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }
    for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
        uint32_t dim = policy->selected_dims[i];
        if (dim >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
            return TKM_ERR;
        }
        out_tokens[i] = breakout_token_quantize(policy, dim, obs[dim]);
    }
    return TKM_OK;
}

static int breakout_token_mlp_init(BreakoutPufferlibTokenPolicy* policy) {
    float w0[TKM_MLP_WINDOW_MAX_INPUT * TKM_MLP_WINDOW_MAX_HIDDEN];
    float b0[TKM_MLP_WINDOW_MAX_HIDDEN];
    float w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float b1[TKM_MLP_WINDOW_MAX_OUTPUT];
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t hidden_dim;

    if (!policy) {
        return TKM_ERR;
    }
    input_dim = breakout_token_mlp_input_dim(policy);
    output_dim = breakout_token_mlp_output_dim(policy);
    hidden_dim = policy->token_mlp_hidden_dim;
    if (input_dim == 0u ||
        input_dim > TKM_MLP_WINDOW_MAX_INPUT ||
        output_dim == 0u ||
        output_dim > TKM_MLP_WINDOW_MAX_OUTPUT ||
        hidden_dim == 0u ||
        hidden_dim > TKM_MLP_WINDOW_MAX_HIDDEN) {
        return TKM_ERR;
    }

    for (uint32_t i = 0u; i < input_dim; i++) {
        for (uint32_t h = 0u; h < hidden_dim; h++) {
            int pattern = (int)(((i + 3u) * (h + 5u)) % 17u) - 8;
            w0[i * hidden_dim + h] = 0.005f * (float)pattern;
        }
    }
    for (uint32_t h = 0u; h < hidden_dim; h++) {
        b0[h] = 0.02f;
        for (uint32_t output = 0u; output < output_dim; output++) {
            int pattern = (int)(((h + 11u) * (output + 7u)) % 13u) - 6;
            w1[h * output_dim + output] = 0.005f * (float)pattern;
        }
    }
    for (uint32_t output = 0u; output < output_dim; output++) {
        b1[output] = 0.0f;
    }

    if (tkm_mlp_window_init(
            &policy->mlp,
            input_dim,
            hidden_dim,
            output_dim,
            w0,
            b0,
            w1,
            b1) != TKM_OK) {
        return TKM_ERR;
    }
    policy->use_mlp = 1u;
    policy->token_mlp_input_dim = input_dim;
    return TKM_OK;
}

static int breakout_continuous_obs_mlp_init(BreakoutPufferlibTokenPolicy* policy) {
    float w0[TKM_MLP_WINDOW_MAX_INPUT * TKM_MLP_WINDOW_MAX_HIDDEN];
    float b0[TKM_MLP_WINDOW_MAX_HIDDEN];
    float w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float b1[TKM_MLP_WINDOW_MAX_OUTPUT];
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t hidden_dim;

    if (!policy || policy->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        return TKM_ERR;
    }
    input_dim = breakout_token_mlp_input_dim(policy);
    output_dim = policy->selected_dim_count;
    hidden_dim = policy->token_mlp_hidden_dim;
    if (input_dim == 0u ||
        input_dim > TKM_MLP_WINDOW_MAX_INPUT ||
        output_dim == 0u ||
        output_dim > TKM_MLP_WINDOW_MAX_OUTPUT ||
        hidden_dim == 0u ||
        hidden_dim > TKM_MLP_WINDOW_MAX_HIDDEN) {
        return TKM_ERR;
    }

    for (uint32_t i = 0u; i < input_dim; i++) {
        for (uint32_t h = 0u; h < hidden_dim; h++) {
            int pattern = (int)(((i + 13u) * (h + 3u)) % 19u) - 9;
            w0[i * hidden_dim + h] = 0.004f * (float)pattern;
        }
    }
    for (uint32_t h = 0u; h < hidden_dim; h++) {
        b0[h] = 0.01f;
        for (uint32_t output = 0u; output < output_dim; output++) {
            int pattern = (int)(((h + 7u) * (output + 17u)) % 17u) - 8;
            w1[h * output_dim + output] = 0.004f * (float)pattern;
        }
    }
    for (uint32_t output = 0u; output < output_dim; output++) {
        b1[output] = 0.0f;
    }

    if (tkm_mlp_window_init(
            &policy->obs_mlp,
            input_dim,
            hidden_dim,
            output_dim,
            w0,
            b0,
            w1,
            b1) != TKM_OK) {
        return TKM_ERR;
    }
    policy->use_obs_mlp = 1u;
    policy->predicted_obs_ready = 0u;
    policy->predicted_obs_count = output_dim;
    memset(policy->predicted_obs_values, 0, sizeof(policy->predicted_obs_values));
    return TKM_OK;
}

static uint32_t breakout_token_stream_code(uint8_t type, uint16_t token) {
    return ((uint32_t)type << 16u) | (uint32_t)token;
}

static void breakout_token_stream_context_reset(uint32_t* context, uint32_t* context_count) {
    if (!context || !context_count) {
        return;
    }
    for (uint32_t i = 0u; i < BREAKOUT_PUFFERLIB_TOKEN_HISTORY; i++) {
        context[i] = 0u;
    }
    *context_count = 0u;
}

static void breakout_token_stream_context_push(
    uint32_t* context,
    uint32_t* context_count,
    uint32_t history,
    uint32_t code
) {
    uint32_t count;

    if (!context || !context_count || history == 0u) {
        return;
    }
    if (history > BREAKOUT_PUFFERLIB_TOKEN_HISTORY) {
        history = BREAKOUT_PUFFERLIB_TOKEN_HISTORY;
    }
    count = *context_count;
    if (count < history) {
        context[count] = code;
        *context_count = count + 1u;
        return;
    }
    for (uint32_t i = 1u; i < history; i++) {
        context[i - 1u] = context[i];
    }
    context[history - 1u] = code;
}


static void breakout_continuous_history_reset(BreakoutPufferlibTokenPolicy* policy) {
    if (!policy) {
        return;
    }
    memset(policy->value_history, 0, sizeof(policy->value_history));
    policy->value_history_count = 0u;
    memset(policy->action_history, 0, sizeof(policy->action_history));
    policy->action_history_count = 0u;
    memset(policy->predicted_obs_values, 0, sizeof(policy->predicted_obs_values));
    policy->predicted_obs_ready = 0u;
    policy->predicted_obs_count = policy->selected_dim_count;
}

static int breakout_continuous_copy_selected_values(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    float* out_values
) {
    if (!policy || !obs || !out_values ||
        policy->selected_dim_count == 0u ||
        policy->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }
    for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
        uint32_t dim = policy->selected_dims[i];
        if (dim >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
            return TKM_ERR;
        }
        out_values[i] = obs[dim];
    }
    return TKM_OK;
}

static void breakout_continuous_history_push(
    float* history,
    uint32_t* history_count,
    uint32_t history_limit,
    uint32_t selected_dim_count,
    const float* values
) {
    uint32_t stride = BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS;

    if (!history || !history_count || !values || history_limit == 0u || selected_dim_count == 0u) {
        return;
    }
    if (*history_count < history_limit) {
        memcpy(&history[*history_count * stride], values, selected_dim_count * sizeof(float));
        (*history_count)++;
        return;
    }
    for (uint32_t slot = 1u; slot < history_limit; slot++) {
        memcpy(&history[(slot - 1u) * stride], &history[slot * stride], selected_dim_count * sizeof(float));
    }
    memcpy(&history[(history_limit - 1u) * stride], values, selected_dim_count * sizeof(float));
}

static void breakout_action_history_push(
    uint8_t* history,
    uint32_t* history_count,
    uint32_t history_limit,
    uint8_t action
) {
    if (!history || !history_count || history_limit == 0u) {
        return;
    }
    if (*history_count < history_limit) {
        history[*history_count] = action;
        (*history_count)++;
        return;
    }
    for (uint32_t slot = 1u; slot < history_limit; slot++) {
        history[slot - 1u] = history[slot];
    }
    history[history_limit - 1u] = action;
}

static int breakout_continuous_mlp_build_input_from_history(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* history,
    uint32_t history_count,
    const float* obs,
    float* input,
    uint32_t input_capacity
) {
    uint32_t input_dim = breakout_token_mlp_input_dim(policy);
    uint32_t stride = BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS;
    uint32_t selected_dim_count;
    uint32_t pad_count;
    uint32_t input_index = 0u;
    float current[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];

    if (!policy || !history || !obs || !input || input_dim == 0u || input_capacity < input_dim) {
        return TKM_ERR;
    }
    selected_dim_count = policy->selected_dim_count;
    if (selected_dim_count == 0u || selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }
    if (history_count > policy->history) {
        history_count = policy->history;
    }
    pad_count = policy->history - history_count;
    for (uint32_t slot = 0u; slot < policy->history; slot++) {
        for (uint32_t i = 0u; i < selected_dim_count; i++) {
            input[input_index++] = slot < pad_count ? 0.0f : history[(slot - pad_count) * stride + i];
        }
    }
    if (breakout_continuous_copy_selected_values(policy, obs, current) != TKM_OK) {
        return TKM_ERR;
    }
    for (uint32_t i = 0u; i < selected_dim_count; i++) {
        input[input_index++] = current[i];
    }
    return input_index == input_dim ? TKM_OK : TKM_ERR;
}

static float breakout_clamp_selected_obs_value(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t selected_index,
    float value
) {
    uint32_t dim;
    float min_value;
    float max_value;

    if (!policy || selected_index >= policy->selected_dim_count) {
        return value;
    }
    dim = policy->selected_dims[selected_index];
    if (dim >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
        return value;
    }
    min_value = policy->obs_min[dim];
    max_value = policy->obs_max[dim];
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static int breakout_continuous_obs_mlp_train_regression(
    TkmMlpWindow* mlp,
    const float* input,
    const float* target,
    uint32_t target_count,
    float learning_rate
) {
    float pre_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float hidden[TKM_MLP_WINDOW_MAX_HIDDEN];
    float grad_output[TKM_MLP_WINDOW_MAX_OUTPUT];
    float old_w1[TKM_MLP_WINDOW_MAX_HIDDEN * TKM_MLP_WINDOW_MAX_OUTPUT];
    float grad_hidden[TKM_MLP_WINDOW_MAX_HIDDEN];

    if (!mlp || !input || !target ||
        mlp->input_dim == 0u ||
        mlp->hidden_dim == 0u ||
        mlp->output_dim == 0u ||
        target_count != mlp->output_dim ||
        learning_rate <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t h = 0u; h < mlp->hidden_dim; h++) {
        float value = mlp->b0[h];
        for (uint32_t i = 0u; i < mlp->input_dim; i++) {
            value += input[i] * mlp->w0[i * mlp->hidden_dim + h];
        }
        pre_hidden[h] = value;
        hidden[h] = value > 0.0f ? value : 0.0f;
    }

    for (uint32_t h = 0u; h < mlp->hidden_dim; h++) {
        for (uint32_t output = 0u; output < mlp->output_dim; output++) {
            old_w1[h * mlp->output_dim + output] = mlp->w1[h * mlp->output_dim + output];
        }
    }

    for (uint32_t output = 0u; output < mlp->output_dim; output++) {
        float value = mlp->b1[output];
        for (uint32_t h = 0u; h < mlp->hidden_dim; h++) {
            value += hidden[h] * old_w1[h * mlp->output_dim + output];
        }
        grad_output[output] = value - target[output];
    }

    for (uint32_t h = 0u; h < mlp->hidden_dim; h++) {
        float grad = 0.0f;
        for (uint32_t output = 0u; output < mlp->output_dim; output++) {
            grad += grad_output[output] * old_w1[h * mlp->output_dim + output];
            mlp->w1[h * mlp->output_dim + output] -=
                learning_rate * grad_output[output] * hidden[h];
        }
        grad_hidden[h] = pre_hidden[h] > 0.0f ? grad : 0.0f;
        mlp->b0[h] -= learning_rate * grad_hidden[h];
    }

    for (uint32_t output = 0u; output < mlp->output_dim; output++) {
        mlp->b1[output] -= learning_rate * grad_output[output];
    }

    for (uint32_t i = 0u; i < mlp->input_dim; i++) {
        for (uint32_t h = 0u; h < mlp->hidden_dim; h++) {
            mlp->w0[i * mlp->hidden_dim + h] -= learning_rate * grad_hidden[h] * input[i];
        }
    }
    return TKM_OK;
}

static int breakout_continuous_mlp_predict_obs_from_history(
    BreakoutPufferlibTokenPolicy* policy,
    const float* history,
    uint32_t history_count,
    const float* obs
) {
    float input[TKM_MLP_WINDOW_MAX_INPUT];
    float output[TKM_MLP_WINDOW_MAX_OUTPUT];

    if (!policy || !history || !obs || !policy->use_obs_mlp ||
        breakout_continuous_mlp_build_input_from_history(
            policy,
            history,
            history_count,
            obs,
            input,
            TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
        tkm_mlp_window_forward(&policy->obs_mlp, input, output, TKM_MLP_WINDOW_MAX_OUTPUT) != TKM_OK) {
        return TKM_ERR;
    }
    policy->predicted_obs_count = policy->selected_dim_count;
    for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
        policy->predicted_obs_values[i] = breakout_clamp_selected_obs_value(policy, i, output[i]);
    }
    policy->predicted_obs_ready = 1u;
    return TKM_OK;
}

static int breakout_continuous_mlp_predict_action_from_history(
    BreakoutPufferlibTokenPolicy* policy,
    const float* history,
    uint32_t history_count,
    const float* obs,
    uint8_t* out_action
) {
    float input[TKM_MLP_WINDOW_MAX_INPUT];
    float logits[TKM_MLP_WINDOW_MAX_OUTPUT];
    uint32_t best = 0u;

    if (!policy || !obs || !out_action || !policy->use_mlp ||
        breakout_continuous_mlp_build_input_from_history(
            policy,
            history,
            history_count,
            obs,
            input,
            TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
        tkm_mlp_window_forward(&policy->mlp, input, logits, TKM_MLP_WINDOW_MAX_OUTPUT) != TKM_OK) {
        return TKM_ERR;
    }
    for (uint32_t action = 1u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        if (logits[action] > logits[best]) {
            best = action;
        }
    }
    *out_action = (uint8_t)best;
    return TKM_OK;
}

static uint32_t breakout_token_ngram_context_key(
    const uint32_t* context,
    uint32_t context_count
) {
    uint32_t hash = 2166136261u;

    for (uint32_t i = 0u; i < context_count; i++) {
        hash = breakout_token_hash_mix(hash, context[i]);
    }
    return hash == 0u ? 1u : hash;
}

static uint32_t breakout_token_ngram_suffix_offset(
    const uint32_t* context,
    uint32_t context_count,
    uint32_t suffix_count
) {
    (void)context;
    if (suffix_count >= context_count) {
        return 0u;
    }
    return context_count - suffix_count;
}

static BreakoutPufferlibTokenNgramEntry* breakout_token_ngram_entry(
    BreakoutPufferlibTokenPolicy* policy,
    uint32_t key,
    uint8_t create
) {
    uint32_t start;

    if (!policy || key == 0u) {
        return NULL;
    }
    start = key % BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES;
    for (uint32_t probe = 0u; probe < 32u; probe++) {
        uint32_t index = (start + probe) % BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES;
        BreakoutPufferlibTokenNgramEntry* entry = &policy->ngram_entries[index];

        if (entry->used && entry->key == key) {
            return entry;
        }
        if (!entry->used) {
            if (!create) {
                return NULL;
            }
            memset(entry, 0, sizeof(*entry));
            entry->used = 1u;
            entry->key = key;
            return entry;
        }
    }
    if (create) {
        BreakoutPufferlibTokenNgramEntry* entry = &policy->ngram_entries[start];
        memset(entry, 0, sizeof(*entry));
        entry->used = 1u;
        entry->key = key;
        return entry;
    }
    return NULL;
}

static uint16_t breakout_token_obs_class(uint16_t obs_stream_token) {
    return (uint16_t)(TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT + obs_stream_token);
}

static uint8_t breakout_token_class_to_action(uint16_t token_class) {
    if (token_class < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        return (uint8_t)token_class;
    }
    return 0u;
}

static void breakout_token_ngram_update_class(
    BreakoutPufferlibTokenNgramEntry* entry,
    uint16_t token_class
) {
    if (!entry) {
        return;
    }
    for (uint32_t i = 0u; i < BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES; i++) {
        if (entry->counts[i] > 0u && entry->classes[i] == token_class) {
            if (entry->counts[i] < 0xffffu) {
                entry->counts[i]++;
            }
            return;
        }
    }
    for (uint32_t i = 0u; i < BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES; i++) {
        if (entry->counts[i] == 0u) {
            entry->classes[i] = token_class;
            entry->counts[i] = 1u;
            return;
        }
    }
    for (uint32_t i = 0u; i < BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES; i++) {
        entry->counts[i]--;
    }
}

static uint16_t breakout_token_ngram_predict_class(const BreakoutPufferlibTokenNgramEntry* entry) {
    uint32_t best = 0u;

    if (!entry) {
        return 0u;
    }
    for (uint32_t i = 1u; i < BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES; i++) {
        if (entry->counts[i] > entry->counts[best]) {
            best = i;
        }
    }
    return entry->classes[best];
}

static uint32_t breakout_token_ngram_best_class_count(const BreakoutPufferlibTokenNgramEntry* entry) {
    uint32_t best = 0u;

    if (!entry) {
        return 0u;
    }
    for (uint32_t i = 1u; i < BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES; i++) {
        if (entry->counts[i] > entry->counts[best]) {
            best = i;
        }
    }
    return entry->counts[best];
}

static uint16_t breakout_token_ngram_predict_class_with_prior(
    const BreakoutPufferlibTokenPolicy* policy,
    const BreakoutPufferlibTokenNgramEntry* entry
) {
    uint16_t best = breakout_token_ngram_predict_class(entry);
    uint32_t best_count = breakout_token_ngram_best_class_count(entry);
    uint32_t output_dim;

    if (!policy || best_count > 0u) {
        return best;
    }
    output_dim = breakout_token_mlp_output_dim(policy);
    best = 0u;
    for (uint32_t token_class = 1u; token_class < output_dim; token_class++) {
        if (policy->ngram_class_prior[token_class] > policy->ngram_class_prior[best]) {
            best = (uint16_t)token_class;
        }
    }
    return best;
}

static void breakout_token_ngram_update_class_contexts(
    BreakoutPufferlibTokenPolicy* policy,
    const uint32_t* context,
    uint32_t context_count,
    uint16_t token_class
) {
    uint32_t min_count;
    uint32_t count;

    if (!policy || !context) {
        return;
    }
    min_count = policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM ?
        0u : context_count;
    count = context_count;
    for (;;) {
        uint32_t offset = breakout_token_ngram_suffix_offset(context, context_count, count);
        uint32_t key = breakout_token_ngram_context_key(&context[offset], count);
        BreakoutPufferlibTokenNgramEntry* entry = breakout_token_ngram_entry(policy, key, 1u);

        breakout_token_ngram_update_class(entry, token_class);
        if (count == min_count) {
            break;
        }
        count--;
    }
}

static uint16_t breakout_token_ngram_predict_class_from_context_with_diag(
    BreakoutPufferlibTokenPolicy* policy,
    const uint32_t* context,
    uint32_t context_count,
    uint32_t* exact_count,
    uint32_t* backoff_count,
    uint32_t* prior_count
) {
    uint32_t key;
    BreakoutPufferlibTokenNgramEntry* entry;

    if (!policy || !context) {
        if (prior_count) {
            (*prior_count)++;
        }
        return 0u;
    }

    key = breakout_token_ngram_context_key(context, context_count);
    entry = breakout_token_ngram_entry(policy, key, 0u);
    if (breakout_token_ngram_best_class_count(entry) > 0u) {
        uint16_t best = breakout_token_ngram_predict_class(entry);
        if (exact_count) {
            (*exact_count)++;
        }
        return best;
    }

    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM) {
        for (uint32_t suffix_count = context_count; suffix_count > 0u; suffix_count--) {
            uint32_t count = suffix_count - 1u;
            uint32_t offset = breakout_token_ngram_suffix_offset(context, context_count, count);

            key = breakout_token_ngram_context_key(&context[offset], count);
            entry = breakout_token_ngram_entry(policy, key, 0u);
            if (breakout_token_ngram_best_class_count(entry) > 0u) {
                uint16_t best = breakout_token_ngram_predict_class(entry);
                if (backoff_count) {
                    (*backoff_count)++;
                }
                return best;
            }
        }
    }

    if (prior_count) {
        (*prior_count)++;
    }
    return breakout_token_ngram_predict_class_with_prior(policy, NULL);
}

static uint16_t breakout_token_obs_stream_token(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t selected_index,
    uint8_t token
) {
    return (uint16_t)(selected_index * policy->bin_count + (uint32_t)token);
}

static int breakout_token_ngram_predict_action_from_context(
    BreakoutPufferlibTokenPolicy* policy,
    const uint32_t* context,
    uint32_t context_count,
    uint8_t* out_action
) {
    uint16_t predicted_class;

    if (!policy || !context || !out_action) {
        return TKM_ERR;
    }
    predicted_class = breakout_token_ngram_predict_class_from_context_with_diag(
        policy, context, context_count, NULL, NULL, NULL);
    *out_action = breakout_token_class_to_action(predicted_class);
    return TKM_OK;
}

static int breakout_token_ngram_pass(
    BreakoutPufferlibTokenPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibTokenConfig* config,
    uint8_t train,
    uint32_t* out_train_rows,
    uint32_t* out_heldout_rows,
    uint32_t* out_obs_correct,
    uint32_t* out_obs_total,
    uint32_t* out_action_correct,
    uint32_t* out_action_total,
    uint32_t* out_exact_predictions,
    uint32_t* out_backoff_predictions,
    uint32_t* out_prior_predictions
) {
    uint32_t context[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
    uint32_t context_count = 0u;
    uint32_t train_rows = 0u;
    uint32_t heldout_rows = 0u;
    uint32_t obs_correct = 0u;
    uint32_t obs_total = 0u;
    uint32_t action_correct = 0u;
    uint32_t action_total = 0u;
    uint32_t exact_predictions = 0u;
    uint32_t backoff_predictions = 0u;
    uint32_t prior_predictions = 0u;
    int32_t prev_env = -1;
    int32_t prev_episode = -1;

    if (!policy || !breakout_dataset_ready(dataset) || !config) {
        return TKM_ERR;
    }
    breakout_token_stream_context_reset(context, &context_count);
    for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        uint8_t obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
        uint8_t heldout;

        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }
        if (row->env_index != prev_env || row->episode != prev_episode) {
            breakout_token_stream_context_reset(context, &context_count);
            breakout_token_stream_context_push(
                context,
                &context_count,
                policy->history,
                breakout_token_stream_code(TKM_SEQUENCE_BOS, 0u));
            prev_env = row->env_index;
            prev_episode = row->episode;
        }
        if (breakout_token_make_selected_tokens(policy, row->obs, obs_tokens) != TKM_OK) {
            return TKM_ERR;
        }
        heldout = breakout_is_heldout(row_index, config->heldout_stride) ? 1u : 0u;
        if (heldout) {
            heldout_rows++;
        } else {
            train_rows++;
        }

        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            uint16_t token = breakout_token_obs_stream_token(policy, i, obs_tokens[i]);
            uint16_t target_class = breakout_token_obs_class(token);

            if (train) {
                if (!heldout) {
                    breakout_token_ngram_update_class_contexts(policy, context, context_count, target_class);
                    policy->ngram_class_prior[target_class]++;
                }
            } else if (heldout) {
                if (breakout_token_ngram_predict_class_from_context_with_diag(
                        policy,
                        context,
                        context_count,
                        NULL,
                        NULL,
                        NULL) == target_class) {
                    obs_correct++;
                }
                obs_total++;
            }
            breakout_token_stream_context_push(
                context,
                &context_count,
                policy->history,
                breakout_token_stream_code(TKM_SEQUENCE_OBS, token));
        }

        {
            if (train) {
                if (!heldout) {
                    breakout_token_ngram_update_class_contexts(policy, context, context_count, row->action);
                    policy->ngram_class_prior[row->action]++;
                }
            } else if (heldout) {
                uint16_t predicted = breakout_token_ngram_predict_class_from_context_with_diag(
                    policy,
                    context,
                    context_count,
                    &exact_predictions,
                    &backoff_predictions,
                    &prior_predictions);

                if (predicted == row->action) {
                    action_correct++;
                }
                action_total++;
            }
            breakout_token_stream_context_push(
                context,
                &context_count,
                policy->history,
                breakout_token_stream_code(TKM_SEQUENCE_ACTION, row->action));
        }
    }

    if (out_train_rows) {
        *out_train_rows = train_rows;
    }
    if (out_heldout_rows) {
        *out_heldout_rows = heldout_rows;
    }
    if (out_obs_correct) {
        *out_obs_correct = obs_correct;
    }
    if (out_obs_total) {
        *out_obs_total = obs_total;
    }
    if (out_action_correct) {
        *out_action_correct = action_correct;
    }
    if (out_action_total) {
        *out_action_total = action_total;
    }
    if (out_exact_predictions) {
        *out_exact_predictions = exact_predictions;
    }
    if (out_backoff_predictions) {
        *out_backoff_predictions = backoff_predictions;
    }
    if (out_prior_predictions) {
        *out_prior_predictions = prior_predictions;
    }
    return TKM_OK;
}

static int breakout_token_build_context_mlp_input(
    const BreakoutPufferlibTokenPolicy* policy,
    const uint32_t* context,
    uint32_t context_count,
    float* input,
    uint32_t input_capacity
) {
    uint32_t input_dim;
    uint32_t pad_count;
    float token_scale;
    uint32_t input_index = 0u;

    if (!policy || !context || !input ||
        policy->history == 0u ||
        policy->history > BREAKOUT_PUFFERLIB_TOKEN_HISTORY) {
        return TKM_ERR;
    }
    input_dim = breakout_token_mlp_input_dim(policy);
    if (input_dim == 0u || input_capacity < input_dim) {
        return TKM_ERR;
    }
    if (context_count > policy->history) {
        context_count = policy->history;
    }
    pad_count = policy->history - context_count;
    token_scale = (float)(policy->selected_dim_count * policy->bin_count);
    if (token_scale < (float)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        token_scale = (float)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
    }
    if (token_scale <= 0.0f) {
        return TKM_ERR;
    }

    for (uint32_t slot = 0u; slot < policy->history; slot++) {
        uint8_t type = TKM_SEQUENCE_PAD;
        uint32_t token = 0u;

        if (slot >= pad_count) {
            uint32_t code = context[slot - pad_count];
            type = (uint8_t)((code >> 16u) & 0xffu);
            token = code & 0xffffu;
        }
        if (type > TKM_SEQUENCE_TRANSITION) {
            type = TKM_SEQUENCE_TRANSITION;
        }
        if ((float)token > token_scale) {
            token = (uint32_t)token_scale;
        }
        input[input_index++] = 2.0f * ((float)type / (float)TKM_SEQUENCE_TRANSITION) - 1.0f;
        input[input_index++] = 2.0f * ((float)token / token_scale) - 1.0f;
    }
    return input_index == input_dim ? TKM_OK : TKM_ERR;
}

static uint16_t breakout_token_mlp_best_class(
    const float* logits,
    uint32_t class_count
) {
    uint16_t best = 0u;

    if (!logits || class_count == 0u) {
        return 0u;
    }
    for (uint32_t i = 1u; i < class_count; i++) {
        if (logits[i] > logits[best]) {
            best = (uint16_t)i;
        }
    }
    return best;
}

static int breakout_token_mlp_predict_class(
    BreakoutPufferlibTokenPolicy* policy,
    const uint32_t* context,
    uint32_t context_count,
    uint16_t* out_class
) {
    float input[TKM_MLP_WINDOW_MAX_INPUT];
    float logits[TKM_MLP_WINDOW_MAX_OUTPUT];
    uint32_t output_dim;

    if (!policy || !context || !out_class) {
        return TKM_ERR;
    }
    output_dim = breakout_token_mlp_output_dim(policy);
    if (output_dim == 0u || output_dim > TKM_MLP_WINDOW_MAX_OUTPUT) {
        return TKM_ERR;
    }
    if (breakout_token_build_context_mlp_input(
            policy,
            context,
            context_count,
            input,
            TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
        tkm_mlp_window_forward(&policy->mlp, input, logits, TKM_MLP_WINDOW_MAX_OUTPUT) != TKM_OK) {
        return TKM_ERR;
    }
    *out_class = breakout_token_mlp_best_class(logits, output_dim);
    return TKM_OK;
}

static int breakout_token_mlp_pass(
    BreakoutPufferlibTokenPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibTokenConfig* config,
    uint8_t train,
    uint32_t* out_train_rows,
    uint32_t* out_heldout_rows,
    uint32_t* out_obs_correct,
    uint32_t* out_obs_total,
    uint32_t* out_action_correct,
    uint32_t* out_action_total
) {
    uint32_t context[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
    uint32_t context_count = 0u;
    uint32_t train_rows = 0u;
    uint32_t heldout_rows = 0u;
    uint32_t obs_correct = 0u;
    uint32_t obs_total = 0u;
    uint32_t action_correct = 0u;
    uint32_t action_total = 0u;
    int32_t prev_env = -1;
    int32_t prev_episode = -1;

    if (!policy || !breakout_dataset_ready(dataset) || !config) {
        return TKM_ERR;
    }
    breakout_token_stream_context_reset(context, &context_count);
    for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        uint8_t obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
        uint8_t heldout;

        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }
        if (row->env_index != prev_env || row->episode != prev_episode) {
            breakout_token_stream_context_reset(context, &context_count);
            breakout_token_stream_context_push(
                context,
                &context_count,
                policy->history,
                breakout_token_stream_code(TKM_SEQUENCE_BOS, 0u));
            prev_env = row->env_index;
            prev_episode = row->episode;
        }
        if (breakout_token_make_selected_tokens(policy, row->obs, obs_tokens) != TKM_OK) {
            return TKM_ERR;
        }
        heldout = breakout_is_heldout(row_index, config->heldout_stride) ? 1u : 0u;
        if (heldout) {
            heldout_rows++;
        } else {
            train_rows++;
        }

        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            uint16_t token = breakout_token_obs_stream_token(policy, i, obs_tokens[i]);
            uint16_t target_class = breakout_token_obs_class(token);

            if (train) {
                if (!heldout) {
                    float input[TKM_MLP_WINDOW_MAX_INPUT];

                    if (breakout_token_build_context_mlp_input(
                            policy,
                            context,
                            context_count,
                            input,
                            TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
                        tkm_mlp_window_train_cross_entropy(
                            &policy->mlp,
                            input,
                            target_class,
                            config->learning_rate) != TKM_OK) {
                        return TKM_ERR;
                    }
                }
            } else if (heldout) {
                uint16_t predicted = 0u;

                if (breakout_token_mlp_predict_class(
                        policy,
                        context,
                        context_count,
                        &predicted) != TKM_OK) {
                    return TKM_ERR;
                }
                if (predicted == target_class) {
                    obs_correct++;
                }
                obs_total++;
            }

            breakout_token_stream_context_push(
                context,
                &context_count,
                policy->history,
                breakout_token_stream_code(TKM_SEQUENCE_OBS, token));
        }

        if (train) {
            if (!heldout) {
                float input[TKM_MLP_WINDOW_MAX_INPUT];

                if (breakout_token_build_context_mlp_input(
                        policy,
                        context,
                        context_count,
                        input,
                        TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
                    tkm_mlp_window_train_cross_entropy(
                        &policy->mlp,
                        input,
                        row->action,
                        config->learning_rate) != TKM_OK) {
                    return TKM_ERR;
                }
            }
        } else if (heldout) {
            uint16_t predicted = 0u;

            if (breakout_token_mlp_predict_class(
                    policy,
                    context,
                    context_count,
                    &predicted) != TKM_OK) {
                return TKM_ERR;
            }
            if (predicted == row->action) {
                action_correct++;
            }
            action_total++;
        }

        breakout_token_stream_context_push(
            context,
            &context_count,
            policy->history,
            breakout_token_stream_code(TKM_SEQUENCE_ACTION, row->action));
    }

    if (out_train_rows) {
        *out_train_rows = train_rows;
    }
    if (out_heldout_rows) {
        *out_heldout_rows = heldout_rows;
    }
    if (out_obs_correct) {
        *out_obs_correct = obs_correct;
    }
    if (out_obs_total) {
        *out_obs_total = obs_total;
    }
    if (out_action_correct) {
        *out_action_correct = action_correct;
    }
    if (out_action_total) {
        *out_action_total = action_total;
    }
    return TKM_OK;
}


static int breakout_continuous_mlp_pass(
    BreakoutPufferlibTokenPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibTokenConfig* config,
    uint8_t train,
    uint32_t* out_train_rows,
    uint32_t* out_heldout_rows,
    uint32_t* out_action_correct,
    uint32_t* out_action_total
) {
    float history[BREAKOUT_PUFFERLIB_TOKEN_HISTORY * BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    uint32_t history_count = 0u;
    uint32_t train_rows = 0u;
    uint32_t heldout_rows = 0u;
    uint32_t action_correct = 0u;
    uint32_t action_total = 0u;
    uint32_t last_episode = 0xffffffffu;

    if (!policy || !breakout_dataset_ready(dataset) || !config) {
        return TKM_ERR;
    }
    memset(history, 0, sizeof(history));
    for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        float input[TKM_MLP_WINDOW_MAX_INPUT];
        float selected_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
        uint8_t heldout = breakout_is_heldout(row_index, config->heldout_stride) ? 1u : 0u;
        uint32_t row_episode;
        const TkmFloatTransition* next_row = NULL;
        uint8_t has_next_obs_target = 0u;

        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }
        if (row_index + 1u < dataset->row_count && !row->terminal) {
            const TkmFloatTransition* candidate = &dataset->rows[row_index + 1u];
            if (breakout_transition_row_ready(candidate) &&
                candidate->env_index == row->env_index &&
                candidate->episode == row->episode) {
                next_row = candidate;
                has_next_obs_target = 1u;
            }
        }
        row_episode = row->episode < 0 ? 0u : (uint32_t)row->episode;
        if (row_index == 0u || row_episode != last_episode || row->t == 0) {
            memset(history, 0, sizeof(history));
            history_count = 0u;
            last_episode = row_episode;
        }
        if (train && heldout) {
            heldout_rows++;
        } else if (train) {
            if (breakout_continuous_mlp_build_input_from_history(
                    policy,
                    history,
                    history_count,
                    row->obs,
                    input,
                    TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
                tkm_mlp_window_train_cross_entropy(
                    &policy->mlp,
                    input,
                    (uint16_t)row->action,
                    config->learning_rate) != TKM_OK) {
                return TKM_ERR;
            }
            if (has_next_obs_target) {
                float target_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];

                if (breakout_continuous_copy_selected_values(policy, next_row->obs, target_values) != TKM_OK ||
                    breakout_continuous_obs_mlp_train_regression(
                        &policy->obs_mlp,
                        input,
                        target_values,
                        policy->selected_dim_count,
                        config->learning_rate) != TKM_OK) {
                    return TKM_ERR;
                }
            }
            train_rows++;
        } else if (heldout) {
            uint8_t predicted = 0u;

            if (breakout_continuous_mlp_predict_action_from_history(
                    policy,
                    history,
                    history_count,
                    row->obs,
                    &predicted) != TKM_OK) {
                return TKM_ERR;
            }
            if (predicted == row->action) {
                action_correct++;
            }
            action_total++;
            heldout_rows++;
        } else {
            train_rows++;
        }
        if (breakout_continuous_copy_selected_values(policy, row->obs, selected_values) != TKM_OK) {
            return TKM_ERR;
        }
        breakout_continuous_history_push(
            history,
            &history_count,
            policy->history,
            policy->selected_dim_count,
            selected_values);
        if (row->terminal) {
            memset(history, 0, sizeof(history));
            history_count = 0u;
        }
    }
    if (out_train_rows) {
        *out_train_rows = train_rows;
    }
    if (out_heldout_rows) {
        *out_heldout_rows = heldout_rows;
    }
    if (out_action_correct) {
        *out_action_correct = action_correct;
    }
    if (out_action_total) {
        *out_action_total = action_total;
    }
    return TKM_OK;
}

static int breakout_token_select_dims(
    BreakoutPufferlibTokenPolicy* policy,
    const BreakoutPufferlibTokenConfig* config
) {
    BreakoutPufferlibTokenObservationSelectionKind selection_kind;

    if (!policy || !config) {
        return TKM_ERR;
    }
    selection_kind = config->observation_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_UNSPECIFIED ?
        BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS : config->observation_selection_kind;

    if (selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS) {
        if (policy->selected_dim_count > TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
            return TKM_ERR;
        }
        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            policy->selected_dims[i] = i;
        }
        return TKM_OK;
    }
    if (selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL) {
        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            if (config->manual_selected_dims[i] >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
                return TKM_ERR;
            }
            policy->selected_dims[i] = config->manual_selected_dims[i];
        }
        return TKM_OK;
    }
    return TKM_ERR;
}

static int breakout_token_policy_ready(const BreakoutPufferlibTokenPolicy* policy) {
    if (!policy) {
        return 0;
    }
    return policy &&
        policy->bin_count > 0u &&
        policy->bin_count <= BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS &&
        policy->selected_dim_count > 0u &&
        policy->selected_dim_count <= BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS &&
        policy->history <= BREAKOUT_PUFFERLIB_TOKEN_HISTORY &&
        policy->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_UNSPECIFIED &&
        policy->head_kind == BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL;
}

static int breakout_token_report_copy_layer_names(
    const BreakoutPufferlibTokenPolicy* policy,
    const BreakoutPufferlibTokenConfig* config,
    BreakoutPufferlibTokenReport* report
) {
    BreakoutPufferlibTokenObservationSelectionKind selection_kind;

    if (!policy || !config || !report) {
        return TKM_ERR;
    }
    selection_kind = config->observation_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_UNSPECIFIED ?
        BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS : config->observation_selection_kind;
    if (breakout_copy_name(
            report->sequence_layout_name,
            BREAKOUT_PUFFERLIB_LAYER_NAME_MAX,
            breakout_token_layout_name(policy->model_kind)) != TKM_OK ||
        breakout_copy_name(
            report->observation_tokenizer_name,
            BREAKOUT_PUFFERLIB_LAYER_NAME_MAX,
            breakout_token_observation_tokenizer_name(policy->model_kind)) != TKM_OK ||
        breakout_copy_name(
            report->observation_selection_name,
            BREAKOUT_PUFFERLIB_LAYER_NAME_MAX,
            breakout_token_selection_name(selection_kind)) != TKM_OK ||
        breakout_copy_name(
            report->input_feature_name,
            BREAKOUT_PUFFERLIB_LAYER_NAME_MAX,
            breakout_token_input_feature_name(policy->input_feature_kind)) != TKM_OK ||
        breakout_copy_name(
            report->sequence_model_name,
            BREAKOUT_PUFFERLIB_LAYER_NAME_MAX,
            breakout_token_model_name(policy->model_kind)) != TKM_OK ||
        breakout_copy_name(
            report->action_head_name,
            BREAKOUT_PUFFERLIB_LAYER_NAME_MAX,
            breakout_token_head_name(policy->head_kind)) != TKM_OK) {
        return TKM_ERR;
    }
    return TKM_OK;
}

int breakout_pufferlib_token_config_from_ini(
    const TkmIni* ini,
    BreakoutPufferlibTokenConfig* out
) {
    const char* layout_kind;
    const char* layout_predict;
    const char* tokenizer_kind;
    const char* tokenizer_selection;
    const char* action_tokenizer_kind;
    const char* input_feature_kind;
    const char* model_kind;
    const char* head_kind;
    int32_t parsed_i32 = 0;
    float parsed_f32 = 0.0f;

    if (!ini || !out) {
        return TKM_ERR;
    }
    memset(out, 0, sizeof(*out));

    layout_kind = tkm_ini_get(ini, "sequence_layout", "kind", "autoregressive_obs_action");
    layout_predict = tkm_ini_get(ini, "sequence_layout", "predict", "next_token");
    tokenizer_kind = tkm_ini_get(ini, "tokenizer.observation", "kind", "selected_quantized");
    tokenizer_selection = tkm_ini_get(ini, "tokenizer.observation", "selection", "first_dims");
    action_tokenizer_kind = tkm_ini_get(ini, "tokenizer.action", "kind", "categorical");
    input_feature_kind = tkm_ini_get(ini, "input_features", "kind", "token_stream");
    model_kind = tkm_ini_get(ini, "sequence_model", "kind", "token_ngram");
    head_kind = tkm_ini_get(ini, "action_head", "kind", "categorical");

    if (breakout_token_model_kind_from_string(model_kind, &out->model_kind) != TKM_OK ||
        breakout_token_head_kind_from_string(head_kind, &out->head_kind) != TKM_OK ||
        breakout_token_selection_kind_from_string(
            tokenizer_selection,
            &out->observation_selection_kind) != TKM_OK ||
        breakout_token_input_feature_kind_from_string(
            input_feature_kind,
            &out->input_feature_kind) != TKM_OK) {
        return TKM_ERR;
    }
    if (breakout_token_model_predicts_next_token(out->model_kind)) {
        if (strcmp(layout_kind, "autoregressive_obs_action") != 0 ||
            strcmp(layout_predict, "next_token") != 0 ||
            strcmp(tokenizer_kind, "selected_quantized") != 0 ||
            strcmp(action_tokenizer_kind, "categorical") != 0 ||
            out->head_kind != BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL ||
            out->input_feature_kind != BREAKOUT_PUFFERLIB_TOKEN_INPUT_TOKEN_STREAM) {
            return TKM_ERR;
        }
    } else if (breakout_token_model_predicts_action(out->model_kind)) {
        if (strcmp(layout_kind, "obs_action_interleaved") != 0 ||
            strcmp(layout_predict, "action") != 0 ||
            strcmp(tokenizer_kind, "selected_continuous") != 0 ||
            strcmp(action_tokenizer_kind, "categorical") != 0 ||
            out->head_kind != BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL ||
            out->input_feature_kind != BREAKOUT_PUFFERLIB_TOKEN_INPUT_VALUE_WINDOW) {
            return TKM_ERR;
        }
    } else {
        return TKM_ERR;
    }

    if (tkm_ini_get_i32(ini, "tokenizer.observation", "bins", 16, &parsed_i32) != TKM_OK ||
        parsed_i32 <= 1 ||
        parsed_i32 > (int32_t)BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS) {
        return TKM_ERR;
    }
    out->bin_count = (uint32_t)parsed_i32;

    if (tkm_ini_get_i32(ini, "tokenizer.observation", "dims", 16, &parsed_i32) != TKM_OK ||
        parsed_i32 <= 0 ||
        parsed_i32 > (int32_t)BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }
    out->selected_dim_count = (uint32_t)parsed_i32;

    if (out->observation_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL) {
        const char* indices = tkm_ini_get(ini, "tokenizer.observation", "indices", "");

        if (breakout_token_parse_manual_selected_dims(
                indices,
                out->selected_dim_count,
                out->manual_selected_dims) != TKM_OK) {
            return TKM_ERR;
        }
    }


    if (tkm_ini_get_i32(ini, "sequence_layout", "history", 8, &parsed_i32) != TKM_OK ||
        parsed_i32 < 0 ||
        parsed_i32 > (int32_t)BREAKOUT_PUFFERLIB_TOKEN_HISTORY) {
        return TKM_ERR;
    }
    out->history = (uint32_t)parsed_i32;

    if (tkm_ini_get_i32(
            ini,
            "sequence_model",
            "hidden",
            BREAKOUT_PUFFERLIB_TOKEN_DEFAULT_MLP_HIDDEN,
            &parsed_i32) != TKM_OK ||
        parsed_i32 <= 0 ||
        parsed_i32 > (int32_t)TKM_MLP_WINDOW_MAX_HIDDEN) {
        return TKM_ERR;
    }
    out->model_hidden_dim = (uint32_t)parsed_i32;

    if (tkm_ini_get_i32(ini, "sequence_model", "epochs", 80, &parsed_i32) != TKM_OK ||
        parsed_i32 <= 0) {
        return TKM_ERR;
    }
    out->epochs = (uint32_t)parsed_i32;

    if (tkm_ini_get_f32(ini, "sequence_model", "learning_rate", 0.01f, &parsed_f32) != TKM_OK ||
        parsed_f32 <= 0.0f) {
        return TKM_ERR;
    }
    out->learning_rate = parsed_f32;

    if (tkm_ini_get_i32(ini, "action_head", "actions", 3, &parsed_i32) != TKM_OK ||
        parsed_i32 != (int32_t)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        return TKM_ERR;
    }
    if (tkm_ini_get_i32(ini, "tokenizer.action", "actions", 3, &parsed_i32) != TKM_OK ||
        parsed_i32 != (int32_t)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) {
        return TKM_ERR;
    }

    out->heldout_stride = 5u;
    out->use_pair_features = 1u;
    return TKM_OK;
}

int breakout_pufferlib_token_policy_train(
    BreakoutPufferlibTokenPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibTokenConfig* config,
    BreakoutPufferlibTokenReport* report
) {
    uint32_t train_rows = 0u;
    uint32_t heldout_rows = 0u;

    if (!policy ||
        !breakout_dataset_ready(dataset) ||
        !config ||
        !report ||
        config->bin_count == 0u ||
        config->bin_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS ||
        config->selected_dim_count == 0u ||
        config->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS ||
        config->epochs == 0u ||
        config->learning_rate <= 0.0f ||
        config->heldout_stride < 2u ||
        config->bin_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS ||
        config->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS ||
        config->history > BREAKOUT_PUFFERLIB_TOKEN_HISTORY ||
        (config->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_UNSPECIFIED &&
            config->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW &&
            config->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM &&
            config->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM &&
            config->model_kind != BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) ||
        (config->head_kind != BREAKOUT_PUFFERLIB_TOKEN_HEAD_UNSPECIFIED &&
            config->head_kind != BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL) ||
        (config->input_feature_kind != BREAKOUT_PUFFERLIB_TOKEN_INPUT_UNSPECIFIED &&
            config->input_feature_kind != BREAKOUT_PUFFERLIB_TOKEN_INPUT_TOKEN_STREAM &&
            config->input_feature_kind != BREAKOUT_PUFFERLIB_TOKEN_INPUT_VALUE_WINDOW)) {
        return TKM_ERR;
    }
    memset(report, 0, sizeof(*report));

    memset(policy, 0, sizeof(*policy));
    policy->bin_count = config->bin_count;
    policy->selected_dim_count = config->selected_dim_count;
    policy->use_pair_features = config->use_pair_features ? 1u : 0u;
    policy->history = config->history > 0u ? config->history : BREAKOUT_PUFFERLIB_TOKEN_HISTORY;
    policy->model_kind = config->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_UNSPECIFIED ?
        BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM : config->model_kind;
    policy->head_kind = config->head_kind == BREAKOUT_PUFFERLIB_TOKEN_HEAD_UNSPECIFIED ?
        BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL : config->head_kind;
    policy->input_feature_kind =
        config->input_feature_kind == BREAKOUT_PUFFERLIB_TOKEN_INPUT_UNSPECIFIED ?
            BREAKOUT_PUFFERLIB_TOKEN_INPUT_TOKEN_STREAM : config->input_feature_kind;
    policy->use_mlp =
        (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW ||
            policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) ? 1u : 0u;
    policy->token_mlp_hidden_dim = config->model_hidden_dim > 0u ?
        config->model_hidden_dim : BREAKOUT_PUFFERLIB_TOKEN_DEFAULT_MLP_HIDDEN;

    for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
        policy->obs_min[dim] = dataset->rows[0].obs[dim];
        policy->obs_max[dim] = dataset->rows[0].obs[dim];
    }


    for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }
        for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
            float value = row->obs[dim];
            if (value < policy->obs_min[dim]) {
                policy->obs_min[dim] = value;
            }
            if (value > policy->obs_max[dim]) {
                policy->obs_max[dim] = value;
            }
        }
    }

    if (breakout_token_select_dims(policy, config) != TKM_OK) {
        return TKM_ERR;
    }
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW) {
        uint32_t action_correct = 0u;
        uint32_t action_total = 0u;

        if (breakout_token_mlp_init(policy) != TKM_OK ||
            breakout_continuous_obs_mlp_init(policy) != TKM_OK) {
            return TKM_ERR;
        }
        for (uint32_t epoch = 0u; epoch < config->epochs; epoch++) {
            if (breakout_continuous_mlp_pass(
                    policy,
                    dataset,
                    config,
                    1u,
                    &train_rows,
                    &heldout_rows,
                    NULL,
                    NULL) != TKM_OK) {
                return TKM_ERR;
            }
        }
        if (breakout_continuous_mlp_pass(
                policy,
                dataset,
                config,
                0u,
                &train_rows,
                &heldout_rows,
                &action_correct,
                &action_total) != TKM_OK ||
            train_rows == 0u ||
            heldout_rows == 0u ||
            action_total == 0u) {
            return TKM_ERR;
        }

        breakout_pufferlib_token_policy_reset(policy);
        report->source_rows = dataset->row_count;
        report->train_rows = train_rows;
        report->heldout_rows = heldout_rows;
        report->heldout_accuracy = (float)action_correct / (float)action_total;
        report->obs_token_accuracy = 0.0f;
        report->action_token_accuracy = report->heldout_accuracy;
        report->bin_count = policy->bin_count;
        report->selected_dim_count = policy->selected_dim_count;
        report->exact_predictions = 0u;
        report->backoff_predictions = 0u;
        report->prior_predictions = 0u;
        if (breakout_token_report_copy_layer_names(policy, config, report) != TKM_OK) {
            return TKM_ERR;
        }
        return TKM_OK;
    }
    if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) {
        uint32_t obs_correct = 0u;
        uint32_t obs_total = 0u;
        uint32_t action_correct = 0u;
        uint32_t action_total = 0u;

        if (breakout_token_mlp_init(policy) != TKM_OK) {
            return TKM_ERR;
        }
        for (uint32_t epoch = 0u; epoch < config->epochs; epoch++) {
            if (breakout_token_mlp_pass(
                    policy,
                    dataset,
                    config,
                    1u,
                    &train_rows,
                    &heldout_rows,
                    NULL,
                    NULL,
                    NULL,
                    NULL) != TKM_OK) {
                return TKM_ERR;
            }
        }
        if (breakout_token_mlp_pass(
                policy,
                dataset,
                config,
                0u,
                &train_rows,
                &heldout_rows,
                &obs_correct,
                &obs_total,
                &action_correct,
                &action_total) != TKM_OK ||
            train_rows == 0u ||
            heldout_rows == 0u ||
            obs_total == 0u ||
            action_total == 0u) {
            return TKM_ERR;
        }

        breakout_pufferlib_token_policy_reset(policy);
        report->source_rows = dataset->row_count;
        report->train_rows = train_rows;
        report->heldout_rows = heldout_rows;
        report->heldout_accuracy = (float)action_correct / (float)action_total;
        report->obs_token_accuracy = (float)obs_correct / (float)obs_total;
        report->action_token_accuracy = (float)action_correct / (float)action_total;
        report->bin_count = policy->bin_count;
        report->selected_dim_count = policy->selected_dim_count;
        report->exact_predictions = 0u;
        report->backoff_predictions = 0u;
        report->prior_predictions = 0u;
        if (breakout_token_report_copy_layer_names(policy, config, report) != TKM_OK) {
            return TKM_ERR;
        }
        return TKM_OK;
    }
    if (breakout_token_model_uses_ngram(policy->model_kind)) {
        uint32_t obs_correct = 0u;
        uint32_t obs_total = 0u;
        uint32_t action_correct = 0u;
        uint32_t action_total = 0u;
        uint32_t exact_predictions = 0u;
        uint32_t backoff_predictions = 0u;
        uint32_t prior_predictions = 0u;

        memset(policy->ngram_entries, 0, sizeof(policy->ngram_entries));
        for (uint32_t epoch = 0u; epoch < config->epochs; epoch++) {
            if (breakout_token_ngram_pass(
                    policy,
                    dataset,
                    config,
                    1u,
                    &train_rows,
                    &heldout_rows,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL) != TKM_OK) {
                return TKM_ERR;
            }
        }
        if (breakout_token_ngram_pass(
                policy,
                dataset,
                config,
                0u,
                &train_rows,
                &heldout_rows,
                &obs_correct,
                &obs_total,
                &action_correct,
                &action_total,
                &exact_predictions,
                &backoff_predictions,
                &prior_predictions) != TKM_OK ||
            train_rows == 0u ||
            heldout_rows == 0u ||
            obs_total == 0u ||
            action_total == 0u) {
            return TKM_ERR;
        }

        breakout_pufferlib_token_policy_reset(policy);
        report->source_rows = dataset->row_count;
        report->train_rows = train_rows;
        report->heldout_rows = heldout_rows;
        report->heldout_accuracy = (float)action_correct / (float)action_total;
        report->obs_token_accuracy = (float)obs_correct / (float)obs_total;
        report->action_token_accuracy = (float)action_correct / (float)action_total;
        report->bin_count = policy->bin_count;
        report->selected_dim_count = policy->selected_dim_count;
        report->exact_predictions = exact_predictions;
        report->backoff_predictions = backoff_predictions;
        report->prior_predictions = prior_predictions;
        if (breakout_token_report_copy_layer_names(policy, config, report) != TKM_OK) {
            return TKM_ERR;
        }
        return TKM_OK;
    }
    return TKM_ERR;
}

int breakout_pufferlib_token_policy_reset(BreakoutPufferlibTokenPolicy* policy) {
    if (!breakout_token_policy_ready(policy)) {
        return TKM_ERR;
    }
    breakout_token_stream_context_reset(policy->stream_context, &policy->stream_context_count);
    breakout_continuous_history_reset(policy);
    if (breakout_token_model_uses_token_stream(policy->model_kind)) {
        breakout_token_stream_context_push(
            policy->stream_context,
            &policy->stream_context_count,
            policy->history,
            breakout_token_stream_code(TKM_SEQUENCE_BOS, 0u));
    }
    return TKM_OK;
}

int breakout_pufferlib_token_policy_predict(
    BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    uint8_t* out_action
) {
    uint8_t current_obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];

    if (!breakout_token_policy_ready(policy) || !obs || !out_action) {
        return TKM_ERR;
    }
    if (breakout_token_model_uses_value_window(policy->model_kind)) {
        float selected_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];

        if (breakout_continuous_mlp_predict_action_from_history(
                policy,
                policy->value_history,
                policy->value_history_count,
                obs,
                out_action) != TKM_OK ||
            breakout_continuous_mlp_predict_obs_from_history(
                policy,
                policy->value_history,
                policy->value_history_count,
                obs) != TKM_OK ||
            breakout_continuous_copy_selected_values(policy, obs, selected_values) != TKM_OK) {
            return TKM_ERR;
        }
        breakout_continuous_history_push(
            policy->value_history,
            &policy->value_history_count,
            policy->history,
            policy->selected_dim_count,
            selected_values);
        breakout_action_history_push(
            policy->action_history,
            &policy->action_history_count,
            policy->history,
            *out_action);
        return TKM_OK;
    }
    if (breakout_token_make_selected_tokens(policy, obs, current_obs_tokens) != TKM_OK) {
        return TKM_ERR;
    }
    if (breakout_token_model_uses_token_stream(policy->model_kind)) {
        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            uint16_t token = breakout_token_obs_stream_token(policy, i, current_obs_tokens[i]);
            breakout_token_stream_context_push(
                policy->stream_context,
                &policy->stream_context_count,
                policy->history,
                breakout_token_stream_code(TKM_SEQUENCE_OBS, token));
        }
        if (breakout_token_model_uses_ngram(policy->model_kind)) {
            if (breakout_token_ngram_predict_action_from_context(
                    policy,
                    policy->stream_context,
                    policy->stream_context_count,
                    out_action) != TKM_OK) {
                return TKM_ERR;
            }
        } else if (policy->model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW) {
            uint16_t predicted_class = 0u;

            if (breakout_token_mlp_predict_class(
                    policy,
                    policy->stream_context,
                    policy->stream_context_count,
                    &predicted_class) != TKM_OK) {
                return TKM_ERR;
            }
            *out_action = breakout_token_class_to_action(predicted_class);
        } else {
            return TKM_ERR;
        }
        breakout_token_stream_context_push(
            policy->stream_context,
            &policy->stream_context_count,
            policy->history,
            breakout_token_stream_code(TKM_SEQUENCE_ACTION, *out_action));
        return TKM_OK;
    }
    return TKM_ERR;
}
