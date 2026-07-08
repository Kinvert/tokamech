#ifndef BREAKOUT_PUFFERLIB_POLICY_H
#define BREAKOUT_PUFFERLIB_POLICY_H

#include <stdint.h>

#include "core/config/ini.h"
#include "core/common/status.h"
#include "core/dataset/float_transition.h"
#include "core/model/mlp_window.h"

#define BREAKOUT_PUFFERLIB_LAYER_NAME_MAX 32u
#define BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS 32u
#define BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS 24u
#define BREAKOUT_PUFFERLIB_TOKEN_HISTORY 8u
#define BREAKOUT_PUFFERLIB_TOKEN_FEATURES 65536u
#define BREAKOUT_PUFFERLIB_TOKEN_DEFAULT_MLP_HIDDEN 64u
#define BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES 4096u
#define BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES 8u

typedef enum {
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_UNSPECIFIED = 0,
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM = 3,
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM = 4,
    BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW = 6
} BreakoutPufferlibTokenModelKind;

typedef enum {
    BREAKOUT_PUFFERLIB_TOKEN_HEAD_UNSPECIFIED = 0,
    BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL = 1
} BreakoutPufferlibTokenHeadKind;

typedef enum {
    BREAKOUT_PUFFERLIB_TOKEN_SELECTION_UNSPECIFIED = 0,
    BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS = 1,
    BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL = 2
} BreakoutPufferlibTokenObservationSelectionKind;

typedef enum {
    BREAKOUT_PUFFERLIB_TOKEN_INPUT_UNSPECIFIED = 0,
    BREAKOUT_PUFFERLIB_TOKEN_INPUT_TOKEN_STREAM = 1
} BreakoutPufferlibTokenInputFeatureKind;

typedef struct {
    uint32_t key;
    uint16_t classes[BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES];
    uint16_t counts[BREAKOUT_PUFFERLIB_TOKEN_NGRAM_CANDIDATES];
    uint8_t used;
} BreakoutPufferlibTokenNgramEntry;

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
    BreakoutPufferlibTokenObservationSelectionKind observation_selection_kind;
    BreakoutPufferlibTokenInputFeatureKind input_feature_kind;
    uint32_t manual_selected_dims[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
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
    uint32_t exact_predictions;
    uint32_t backoff_predictions;
    uint32_t prior_predictions;
    char sequence_layout_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
    char observation_tokenizer_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
    char observation_selection_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
    char input_feature_name[BREAKOUT_PUFFERLIB_LAYER_NAME_MAX];
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
    BreakoutPufferlibTokenInputFeatureKind input_feature_kind;
    uint32_t token_mlp_input_dim;
    uint32_t token_mlp_hidden_dim;
    uint32_t selected_dims[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    float obs_min[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    float obs_max[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    BreakoutPufferlibTokenNgramEntry ngram_entries[BREAKOUT_PUFFERLIB_TOKEN_NGRAM_ENTRIES];
    uint32_t ngram_class_prior[TKM_MLP_WINDOW_MAX_OUTPUT];
    uint32_t stream_context[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
    uint32_t stream_context_count;
    TkmMlpWindow mlp;
} BreakoutPufferlibTokenPolicy;

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
