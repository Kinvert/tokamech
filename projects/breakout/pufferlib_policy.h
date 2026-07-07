#ifndef BREAKOUT_PUFFERLIB_POLICY_H
#define BREAKOUT_PUFFERLIB_POLICY_H

#include <stdint.h>

#include "core/config/ini.h"
#include "core/common/status.h"
#include "core/dataset/float_transition.h"
#include "core/model/mlp_window.h"

#define BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN 32u
#define BREAKOUT_PUFFERLIB_POLICY_MAX_FILTER_NAME 32u
#define BREAKOUT_PUFFERLIB_LAYER_NAME_MAX 32u
#define BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS 32u
#define BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS 24u
#define BREAKOUT_PUFFERLIB_TOKEN_HISTORY 8u
#define BREAKOUT_PUFFERLIB_TOKEN_FEATURES 65536u
#define BREAKOUT_PUFFERLIB_TOKEN_DEFAULT_MLP_HIDDEN 64u
#define BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES 4096u
#define BREAKOUT_PUFFERLIB_TOKEN_NGRAM_OBS_CANDIDATES 4u

typedef enum {
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_UNSPECIFIED = 0,
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW = 1,
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_LINEAR_POLICY = 2,
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM = 3
} BreakoutPufferlibTokenModelKind;

typedef enum {
    BREAKOUT_PUFFERLIB_TOKEN_HEAD_UNSPECIFIED = 0,
    BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL = 1,
    BREAKOUT_PUFFERLIB_TOKEN_HEAD_TYPED_NEXT_TOKEN = 2
} BreakoutPufferlibTokenHeadKind;

typedef struct {
    uint32_t key;
    uint16_t obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_NGRAM_OBS_CANDIDATES];
    uint16_t obs_counts[BREAKOUT_PUFFERLIB_TOKEN_NGRAM_OBS_CANDIDATES];
    uint16_t action_counts[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    uint8_t used;
} BreakoutPufferlibTokenNgramEntry;

typedef struct {
    uint32_t hidden_dim;
    uint32_t epochs;
    float learning_rate;
    uint32_t heldout_stride;
} BreakoutPufferlibBcConfig;

typedef struct {
    uint32_t source_rows;
    uint32_t train_rows;
    uint32_t heldout_rows;
    float heldout_accuracy;
    char filter_name[BREAKOUT_PUFFERLIB_POLICY_MAX_FILTER_NAME];
} BreakoutPufferlibBcReport;

typedef struct {
    uint32_t obs_dim;
    uint32_t action_count;
    uint32_t hidden_dim;
    TkmMlpWindow mlp;
} BreakoutPufferlibPolicy;

typedef enum {
    BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_FULL_SEARCH = 1,
    BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_NEXT_ROW = 2
} BreakoutPufferlibSequenceMatch;

typedef struct {
    uint32_t cursor;
    uint8_t has_cursor;
    uint32_t reanchor_interval;
    float max_sequence_distance;
} BreakoutPufferlibSequenceCursor;

typedef struct {
    float deadzone;
    float aim_offset;
} BreakoutPufferlibInterceptPolicy;

typedef struct {
    uint32_t source_rows;
    uint32_t train_rows;
    uint32_t heldout_rows;
    float heldout_accuracy;
    float deadzone;
    float aim_offset;
} BreakoutPufferlibInterceptReport;

typedef struct {
    uint32_t bin_count;
    uint32_t selected_dim_count;
    uint32_t epochs;
    float learning_rate;
    uint32_t heldout_stride;
    uint32_t history;
    BreakoutPufferlibTokenModelKind model_kind;
    uint32_t model_hidden_dim;
    BreakoutPufferlibTokenHeadKind head_kind;
    uint8_t use_pair_features;
} BreakoutPufferlibTokenConfig;

typedef struct {
    uint32_t source_rows;
    uint32_t train_rows;
    uint32_t heldout_rows;
    float heldout_accuracy;
    uint32_t bin_count;
    uint32_t selected_dim_count;
    float obs_token_accuracy;
    float action_token_accuracy;
    char sequence_layout_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
    char observation_tokenizer_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
    char sequence_model_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
    char action_head_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
} BreakoutPufferlibTokenReport;

typedef struct {
    uint32_t bin_count;
    uint32_t selected_dim_count;
    uint8_t use_pair_features;
    uint8_t use_mlp;
    uint32_t history;
    BreakoutPufferlibTokenModelKind model_kind;
    BreakoutPufferlibTokenHeadKind head_kind;
    uint32_t token_mlp_input_dim;
    uint32_t token_mlp_hidden_dim;
    uint32_t selected_dims[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    float obs_min[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    float obs_max[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    float weights[BREAKOUT_PUFFERLIB_TOKEN_FEATURES * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    BreakoutPufferlibTokenNgramEntry ngram_entries[BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES];
    uint32_t ngram_action_prior[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    uint32_t stream_context[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
    uint32_t stream_context_count;
    TkmMlpWindow mlp;
    uint8_t prev_actions[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
    uint32_t prev_action_count;
    float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    uint32_t prev_obs_count;
} BreakoutPufferlibTokenPolicy;

int breakout_pufferlib_policy_init(BreakoutPufferlibPolicy* policy, uint32_t hidden_dim);
int breakout_pufferlib_policy_predict(
    const BreakoutPufferlibPolicy* policy,
    const float* obs,
    uint8_t* out_action
);
int breakout_pufferlib_policy_train_bc(
    BreakoutPufferlibPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibBcConfig* config,
    const char* filter_name,
    BreakoutPufferlibBcReport* report
);
int breakout_pufferlib_policy_save(
    const BreakoutPufferlibPolicy* policy,
    const char* config_path,
    const char* params_path
);
int breakout_pufferlib_policy_load(
    BreakoutPufferlibPolicy* policy,
    const char* config_path,
    const char* params_path
);
int breakout_pufferlib_nearest_predict(
    const TkmFloatTransitionDataset* dataset,
    const float* obs,
    uint8_t* out_action
);
int breakout_pufferlib_nearest_predict_range(
    const TkmFloatTransitionDataset* dataset,
    const float* obs,
    uint32_t start_index,
    uint32_t row_count,
    uint8_t* out_action,
    uint32_t* out_index,
    float* out_distance
);
int breakout_pufferlib_sequence_cursor_init(
    BreakoutPufferlibSequenceCursor* cursor,
    uint32_t reanchor_interval,
    float max_sequence_distance
);
void breakout_pufferlib_sequence_cursor_reset(BreakoutPufferlibSequenceCursor* cursor);
int breakout_pufferlib_sequence_cursor_predict(
    const TkmFloatTransitionDataset* dataset,
    BreakoutPufferlibSequenceCursor* cursor,
    const float* obs,
    uint32_t step,
    uint8_t* out_action,
    BreakoutPufferlibSequenceMatch* out_match,
    uint32_t* out_index,
    float* out_distance
);
int breakout_pufferlib_intercept_policy_init(
    BreakoutPufferlibInterceptPolicy* policy,
    float deadzone,
    float aim_offset
);
int breakout_pufferlib_intercept_policy_predict(
    const BreakoutPufferlibInterceptPolicy* policy,
    const float* obs,
    uint8_t* out_action
);
int breakout_pufferlib_intercept_policy_train_grid(
    BreakoutPufferlibInterceptPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    BreakoutPufferlibInterceptReport* report
);
int breakout_pufferlib_token_config_from_ini(
    const TkmIni* ini,
    BreakoutPufferlibTokenConfig* out
);
int breakout_pufferlib_token_policy_train(
    BreakoutPufferlibTokenPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibTokenConfig* config,
    BreakoutPufferlibTokenReport* report
);
int breakout_pufferlib_token_policy_reset(BreakoutPufferlibTokenPolicy* policy);
int breakout_pufferlib_token_policy_predict(
    BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    uint8_t* out_action
);

#endif
