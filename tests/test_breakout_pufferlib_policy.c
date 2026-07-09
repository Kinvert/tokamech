#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config/ini.h"
#include "core/dataset/float_transition.h"
#include "projects/breakout/pufferlib_policy.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void add_token_relation_row(
    TkmFloatTransitionDataset* dataset,
    int32_t t,
    float left_value,
    float right_value
) {
    TkmFloatTransition* row;

    CHECK(dataset->row_count < TKM_FLOAT_TRANSITION_MAX_ROWS);
    row = &dataset->rows[dataset->row_count++];
    memset(row, 0, sizeof(*row));
    row->version = 1;
    strcpy(row->env, "pufferlib_breakout");
    strcpy(row->source, "test");
    row->env_index = 0;
    row->episode = 0;
    row->t = t;
    row->obs_dim = TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
    row->obs[0] = left_value;
    row->obs[1] = right_value;
    row->obs[2] = 0.25f * left_value + 0.125f;
    row->obs[3] = 0.25f * right_value + 0.25f;
    if (left_value < right_value - 0.001f) {
        row->action = 1u;
    } else if (left_value > right_value + 0.001f) {
        row->action = 2u;
    } else {
        row->action = 0u;
    }
}

static void test_breakout_token_policy_trains_next_action_from_quantized_tokens(void) {
    static TkmFloatTransitionDataset dataset;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;
    int32_t t = 0;

    tkm_float_transition_dataset_init(&dataset);
    for (int32_t epoch = 0; epoch < 8; epoch++) {
        for (int32_t a = 0; a < 9; a++) {
            for (int32_t b = 0; b < 9; b++) {
                add_token_relation_row(&dataset, t++, (float)a / 8.0f, (float)b / 8.0f);
            }
        }
    }

    config.bin_count = 9u;
    config.selected_dim_count = 4u;
    config.epochs = 8u;
    config.learning_rate = 0.25f;
    config.heldout_stride = 7u;
    config.history = 8u;
    config.model_kind = BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM;
    config.head_kind = BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL;
    config.observation_selection_kind = BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS;
    config.input_feature_kind = BREAKOUT_PUFFERLIB_TOKEN_INPUT_TOKEN_STREAM;
    config.use_pair_features = 1u;
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(report.source_rows == dataset.row_count);
    CHECK(report.train_rows > 0u);
    CHECK(report.heldout_rows > 0u);
    CHECK(report.heldout_accuracy >= 0.80f);

    obs[0] = 0.125f;
    obs[1] = 0.875f;
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT);

    obs[0] = 0.875f;
    obs[1] = 0.125f;
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT);
}

static void make_token_relation_dataset(TkmFloatTransitionDataset* dataset) {
    int32_t t = 0;

    tkm_float_transition_dataset_init(dataset);
    for (int32_t epoch = 0; epoch < 8; epoch++) {
        for (int32_t a = 0; a < 9; a++) {
            for (int32_t b = 0; b < 9; b++) {
                add_token_relation_row(dataset, t++, (float)a / 8.0f, (float)b / 8.0f);
            }
        }
    }
}

static const char* token_autoregressive_ini_text(void) {
    return
        "[sequence_layout]\n"
        "kind = autoregressive_obs_action\n"
        "history = 8\n"
        "predict = next_token\n"
        "\n"
        "[tokenizer.observation]\n"
        "kind = selected_quantized\n"
        "dims = 4\n"
        "selection = first_dims\n"
        "bins = 4\n"
        "\n"
        "[tokenizer.action]\n"
        "kind = categorical\n"
        "actions = 3\n"
        "\n"
        "[sequence_model]\n"
        "kind = token_ngram\n"
        "epochs = 4\n"
        "learning_rate = 0.10\n"
        "\n"
        "[action_head]\n"
        "kind = categorical\n"
        "actions = 3\n";
}

static void set_autoregressive_pattern_obs(float* obs, uint32_t phase) {
    static const float values[4] = {0.05f, 0.35f, 0.65f, 0.95f};

    memset(obs, 0, sizeof(float) * TKM_PUFFERLIB_BREAKOUT_OBS_DIM);
    obs[0] = values[phase % 4u];
    obs[1] = values[(phase + 1u) % 4u];
    obs[2] = values[(phase + 2u) % 4u];
    obs[3] = values[(phase + 3u) % 4u];
}

static uint8_t autoregressive_pattern_action(uint32_t phase) {
    if ((phase % 4u) == 0u) {
        return 0u;
    }
    if ((phase % 4u) == 1u || (phase % 4u) == 3u) {
        return 1u;
    }
    return 2u;
}

static void make_autoregressive_pattern_dataset(TkmFloatTransitionDataset* dataset) {
    tkm_float_transition_dataset_init(dataset);
    for (int32_t episode = 0; episode < 24; episode++) {
        for (int32_t t = 0; t < 8; t++) {
            uint32_t phase = (uint32_t)((episode + t) % 4);
            TkmFloatTransition* row;

            CHECK(dataset->row_count < TKM_FLOAT_TRANSITION_MAX_ROWS);
            row = &dataset->rows[dataset->row_count++];
            memset(row, 0, sizeof(*row));
            row->version = 1;
            strcpy(row->env, "pufferlib_breakout");
            strcpy(row->source, "test");
            row->env_index = 0;
            row->episode = episode;
            row->t = t;
            row->obs_dim = TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
            set_autoregressive_pattern_obs(row->obs, phase);
            row->action = autoregressive_pattern_action(phase);
            row->terminal = t == 7 ? 1u : 0u;
        }
    }
}

static void test_breakout_token_config_selects_autoregressive_next_token_backend(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;

    CHECK(tkm_ini_parse(&ini, token_autoregressive_ini_text()) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM);
    CHECK(config.head_kind == BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL);
    CHECK(config.selected_dim_count == 4u);
    CHECK(config.bin_count == 4u);

    make_autoregressive_pattern_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(strcmp(report.sequence_layout_name, "autoregressive_obs_action") == 0);
    CHECK(strcmp(report.observation_tokenizer_name, "selected_quantized") == 0);
    CHECK(strcmp(report.observation_selection_name, "first_dims") == 0);
    CHECK(strcmp(report.sequence_model_name, "token_ngram") == 0);
    CHECK(strcmp(report.action_head_name, "categorical") == 0);
    CHECK(policy.model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM);
    CHECK(report.obs_token_accuracy >= 0.70f);
    CHECK(report.action_token_accuracy >= 0.70f);

    set_autoregressive_pattern_obs(obs, 2u);
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == autoregressive_pattern_action(2u));
}

static uint8_t backoff_pattern_action(uint32_t phase) {
    if (phase == 2u) {
        return 1u;
    }
    return 0u;
}

static void set_backoff_pattern_obs(float* obs, uint32_t phase, uint32_t episode) {
    float marker = ((float)(episode % 30u) + 1.0f) / 32.0f;

    if (phase == 2u) {
        set_autoregressive_pattern_obs(obs, 2u);
        return;
    }
    memset(obs, 0, sizeof(float) * TKM_PUFFERLIB_BREAKOUT_OBS_DIM);
    obs[0] = marker;
    obs[1] = 0.5f * marker;
    obs[2] = phase == 0u ? 0.125f : 0.375f + 0.125f * marker;
    obs[3] = phase == 0u ? 0.250f : 0.500f + 0.125f * marker;
}

static void make_backoff_pattern_dataset(TkmFloatTransitionDataset* dataset) {
    static const uint32_t phases[3] = {0u, 1u, 2u};

    tkm_float_transition_dataset_init(dataset);
    for (int32_t episode = 0; episode < 30; episode++) {
        for (int32_t t = 0; t < 3; t++) {
            uint32_t phase = phases[t];
            TkmFloatTransition* row;

            CHECK(dataset->row_count < TKM_FLOAT_TRANSITION_MAX_ROWS);
            row = &dataset->rows[dataset->row_count++];
            memset(row, 0, sizeof(*row));
            row->version = 1;
            strcpy(row->env, "pufferlib_breakout");
            strcpy(row->source, "test");
            row->env_index = 0;
            row->episode = episode;
            row->t = t;
            row->obs_dim = TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
            set_backoff_pattern_obs(row->obs, phase, (uint32_t)episode);
            row->action = backoff_pattern_action(phase);
            row->terminal = t == 2 ? 1u : 0u;
        }
    }
}

static void test_breakout_token_backoff_ngram_uses_suffix_before_prior(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;
    const char* text =
        "[sequence_layout]\n"
        "kind = autoregressive_obs_action\n"
        "history = 8\n"
        "predict = next_token\n"
        "[tokenizer.observation]\n"
        "kind = selected_quantized\n"
        "dims = 4\n"
        "selection = first_dims\n"
        "bins = 32\n"
        "[tokenizer.action]\n"
        "kind = categorical\n"
        "actions = 3\n"
        "[sequence_model]\n"
        "kind = token_backoff_ngram\n"
        "epochs = 2\n"
        "learning_rate = 0.10\n"
        "[action_head]\n"
        "kind = categorical\n"
        "actions = 3\n";

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM);

    make_backoff_pattern_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(strcmp(report.sequence_model_name, "token_backoff_ngram") == 0);
    CHECK(report.prior_predictions < report.heldout_rows);
    CHECK(report.backoff_predictions > 0u);

    set_backoff_pattern_obs(obs, 0u, 0u);
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == backoff_pattern_action(0u));

    set_backoff_pattern_obs(obs, 2u, 0u);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == backoff_pattern_action(2u));
}

static void test_breakout_token_mlp_window_predicts_categoricals(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;
    const char* text =
        "[sequence_layout]\n"
        "kind = autoregressive_obs_action\n"
        "history = 8\n"
        "predict = next_token\n"
        "[tokenizer.observation]\n"
        "kind = selected_quantized\n"
        "dims = 4\n"
        "selection = first_dims\n"
        "bins = 8\n"
        "[tokenizer.action]\n"
        "kind = categorical\n"
        "actions = 3\n"
        "[sequence_model]\n"
        "kind = token_mlp_window\n"
        "hidden = 32\n"
        "epochs = 80\n"
        "learning_rate = 0.03\n"
        "[action_head]\n"
        "kind = categorical\n"
        "actions = 3\n";

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW);
    CHECK(config.head_kind == BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL);

    make_autoregressive_pattern_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(strcmp(report.sequence_layout_name, "autoregressive_obs_action") == 0);
    CHECK(strcmp(report.observation_tokenizer_name, "selected_quantized") == 0);
    CHECK(strcmp(report.observation_selection_name, "first_dims") == 0);
    CHECK(strcmp(report.sequence_model_name, "token_mlp_window") == 0);
    CHECK(strcmp(report.action_head_name, "categorical") == 0);
    CHECK(policy.model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW);
    CHECK(policy.token_mlp_input_dim == 16u);
    CHECK(report.obs_token_accuracy >= 0.55f);
    CHECK(report.action_token_accuracy >= 0.65f);

    set_autoregressive_pattern_obs(obs, 2u);
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == autoregressive_pattern_action(2u));
}

static const char* token_stream_with_selection_ini_text(const char* selection, const char* indices) {
    static char text[1024];

    snprintf(
        text,
        sizeof(text),
        "[sequence_layout]\n"
        "kind = autoregressive_obs_action\n"
        "history = 8\n"
        "predict = next_token\n"
        "[tokenizer.observation]\n"
        "kind = selected_quantized\n"
        "dims = 4\n"
        "selection = %s\n"
        "%s%s"
        "bins = 8\n"
        "[tokenizer.action]\n"
        "kind = categorical\n"
        "actions = 3\n"
        "[sequence_model]\n"
        "kind = token_ngram\n"
        "hidden = 24\n"
        "epochs = 4\n"
        "learning_rate = 0.10\n"
        "[action_head]\n"
        "kind = categorical\n"
        "actions = 3\n",
        selection,
        indices ? "indices = " : "",
        indices ? indices : "");
    return text;
}

static void test_breakout_tokenizer_first_dims_selects_prefix_dims(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;

    CHECK(tkm_ini_parse(&ini, token_stream_with_selection_ini_text("first_dims", NULL)) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.observation_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS);

    make_autoregressive_pattern_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(policy.selected_dims[0] == 0u);
    CHECK(policy.selected_dims[1] == 1u);
    CHECK(policy.selected_dims[2] == 2u);
    CHECK(policy.selected_dims[3] == 3u);
    CHECK(strcmp(report.observation_tokenizer_name, "selected_quantized") == 0);
    CHECK(strcmp(report.observation_selection_name, "first_dims") == 0);
    CHECK(report.heldout_accuracy >= 0.65f);
}

static void test_breakout_tokenizer_manual_selection_uses_ini_indices(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;

    CHECK(tkm_ini_parse(&ini, token_stream_with_selection_ini_text("manual", "3,2,1,0\n")) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.observation_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL);
    CHECK(config.manual_selected_dims[0] == 3u);
    CHECK(config.manual_selected_dims[1] == 2u);
    CHECK(config.manual_selected_dims[2] == 1u);
    CHECK(config.manual_selected_dims[3] == 0u);

    make_autoregressive_pattern_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(policy.selected_dims[0] == 3u);
    CHECK(policy.selected_dims[1] == 2u);
    CHECK(policy.selected_dims[2] == 1u);
    CHECK(policy.selected_dims[3] == 0u);
    CHECK(strcmp(report.observation_selection_name, "manual") == 0);
    CHECK(report.heldout_accuracy >= 0.65f);
}

static void test_breakout_token_mlp_window_uses_categorical_full_vocab_head(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;
    const char* text =
        "[sequence_layout]\n"
        "kind = autoregressive_obs_action\n"
        "history = 8\n"
        "predict = next_token\n"
        "[tokenizer.observation]\n"
        "kind = selected_quantized\n"
        "dims = 4\n"
        "selection = first_dims\n"
        "bins = 8\n"
        "[tokenizer.action]\n"
        "kind = categorical\n"
        "actions = 3\n"
        "[sequence_model]\n"
        "kind = token_mlp_window\n"
        "hidden = 32\n"
        "epochs = 80\n"
        "learning_rate = 0.03\n"
        "[action_head]\n"
        "kind = categorical\n"
        "actions = 3\n";

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.head_kind == BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL);

    make_autoregressive_pattern_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(strcmp(report.sequence_model_name, "token_mlp_window") == 0);
    CHECK(strcmp(report.action_head_name, "categorical") == 0);
    CHECK(report.action_token_accuracy >= 0.50f);

    set_autoregressive_pattern_obs(obs, 2u);
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT);
}

static void make_continuous_action_dataset(TkmFloatTransitionDataset* dataset) {
    tkm_float_transition_dataset_init(dataset);
    for (int32_t episode = 0; episode < 18; episode++) {
        for (int32_t t = 0; t < 12; t++) {
            TkmFloatTransition* row;
            float paddle = (float)((episode + t) % 9) / 8.0f;
            float ball = (float)((episode * 2 + t * 3) % 9) / 8.0f;

            CHECK(dataset->row_count < TKM_FLOAT_TRANSITION_MAX_ROWS);
            row = &dataset->rows[dataset->row_count++];
            memset(row, 0, sizeof(*row));
            row->version = 1;
            strcpy(row->env, "pufferlib_breakout");
            strcpy(row->source, "test");
            row->env_index = 0;
            row->episode = episode;
            row->t = t;
            row->obs_dim = TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
            row->obs[0] = paddle;
            row->obs[1] = ball;
            row->obs[2] = paddle - ball;
            row->obs[3] = ball - paddle;
            row->action = paddle + 0.05f < ball ? 2u : (paddle > ball + 0.05f ? 1u : 0u);
            row->terminal = t == 11 ? 1u : 0u;
        }
    }
}

static void test_breakout_continuous_mlp_first_dims_predicts_actions(void) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;
    const char* text =
        "[sequence_layout]\n"
        "kind = obs_action_interleaved\n"
        "history = 4\n"
        "predict = action\n"
        "[tokenizer.observation]\n"
        "kind = selected_continuous\n"
        "dims = 4\n"
        "selection = first_dims\n"
        "bins = 16\n"
        "[tokenizer.action]\n"
        "kind = categorical\n"
        "actions = 3\n"
        "[input_features]\n"
        "kind = value_window\n"
        "[sequence_model]\n"
        "kind = mlp_window\n"
        "hidden = 32\n"
        "epochs = 80\n"
        "learning_rate = 0.03\n"
        "[action_head]\n"
        "kind = categorical\n"
        "actions = 3\n";

    CHECK(tkm_ini_parse(&ini, text) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.model_kind == BREAKOUT_PUFFERLIB_TOKEN_MODEL_MLP_WINDOW);
    CHECK(config.input_feature_kind == BREAKOUT_PUFFERLIB_TOKEN_INPUT_VALUE_WINDOW);

    make_continuous_action_dataset(&dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(strcmp(report.sequence_layout_name, "obs_action_interleaved") == 0);
    CHECK(strcmp(report.observation_tokenizer_name, "selected_continuous") == 0);
    CHECK(strcmp(report.input_feature_name, "value_window") == 0);
    CHECK(strcmp(report.sequence_model_name, "mlp_window") == 0);
    CHECK(policy.token_mlp_input_dim == 20u);
    CHECK(report.heldout_accuracy >= 0.70f);

    obs[0] = 0.125f;
    obs[1] = 0.875f;
    obs[2] = obs[0] - obs[1];
    obs[3] = obs[1] - obs[0];
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 2u);
    CHECK(policy.predicted_obs_ready == 1u);
    CHECK(policy.predicted_obs_count == config.selected_dim_count);
    for (uint32_t i = 0u; i < policy.predicted_obs_count; i++) {
        uint32_t dim = policy.selected_dims[i];
        CHECK(policy.predicted_obs_values[i] >= policy.obs_min[dim] - 0.001f);
        CHECK(policy.predicted_obs_values[i] <= policy.obs_max[dim] + 0.001f);
    }

    obs[0] = 0.875f;
    obs[1] = 0.125f;
    obs[2] = obs[0] - obs[1];
    obs[3] = obs[1] - obs[0];
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 1u);
    CHECK(policy.predicted_obs_ready == 1u);
    CHECK(policy.predicted_obs_count == config.selected_dim_count);
}

typedef enum {
    TOKEN_MATRIX_DATASET_RELATION = 0,
    TOKEN_MATRIX_DATASET_AUTOREGRESSIVE = 1
} BreakoutTokenMatrixDatasetKind;

typedef struct {
    const char* name;
    const char* layout_kind;
    const char* predict_kind;
    const char* observation_tokenizer_kind;
    const char* observation_selection_kind;
    const char* observation_indices;
    const char* model_kind;
    const char* head_kind;
    BreakoutPufferlibTokenModelKind expected_model_kind;
    BreakoutPufferlibTokenHeadKind expected_head_kind;
    BreakoutPufferlibTokenObservationSelectionKind expected_selection_kind;
    BreakoutTokenMatrixDatasetKind dataset_kind;
    uint32_t expected_manual_first_dim;
} BreakoutTokenMatrixCase;

static const char* token_matrix_ini_text(const BreakoutTokenMatrixCase* test_case) {
    static char text[1536];
    const char* action_tokenizer_section =
        strcmp(test_case->layout_kind, "autoregressive_obs_action") == 0 ?
            "[tokenizer.action]\n"
            "kind = categorical\n"
            "actions = 3\n"
            "\n" :
            "";
    const char* bins_line =
        strcmp(test_case->observation_tokenizer_kind, "selected_quantized") == 0 ?
            "bins = 8\n" :
            "";
    const char* indices_key =
        test_case->observation_indices ? "indices = " : "";
    const char* indices_value =
        test_case->observation_indices ? test_case->observation_indices : "";

    snprintf(
        text,
        sizeof(text),
        "[sequence_layout]\n"
        "kind = %s\n"
        "history = 8\n"
        "predict = %s\n"
        "\n"
        "[tokenizer.observation]\n"
        "kind = %s\n"
        "dims = 4\n"
        "selection = %s\n"
        "%s%s"
        "%s"
        "\n"
        "%s"
        "[sequence_model]\n"
        "kind = %s\n"
        "hidden = 24\n"
        "epochs = 10\n"
        "learning_rate = 0.05\n"
        "\n"
        "[action_head]\n"
        "kind = %s\n"
        "actions = 3\n",
        test_case->layout_kind,
        test_case->predict_kind,
        test_case->observation_tokenizer_kind,
        test_case->observation_selection_kind,
        indices_key,
        indices_value,
        bins_line,
        action_tokenizer_section,
        test_case->model_kind,
        test_case->head_kind);
    return text;
}

static void make_token_matrix_dataset(
    BreakoutTokenMatrixDatasetKind kind,
    TkmFloatTransitionDataset* dataset
) {
    if (kind == TOKEN_MATRIX_DATASET_RELATION) {
        make_token_relation_dataset(dataset);
        return;
    }
    make_autoregressive_pattern_dataset(dataset);
}

static void check_token_matrix_case(const BreakoutTokenMatrixCase* test_case) {
    static TkmFloatTransitionDataset dataset;
    TkmIni ini;
    BreakoutPufferlibTokenPolicy policy;
    BreakoutPufferlibTokenConfig config;
    BreakoutPufferlibTokenReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99u;

    CHECK(tkm_ini_parse(&ini, token_matrix_ini_text(test_case)) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_OK);
    CHECK(config.model_kind == test_case->expected_model_kind);
    CHECK(config.head_kind == test_case->expected_head_kind);
    CHECK(config.observation_selection_kind == test_case->expected_selection_kind);
    CHECK(config.selected_dim_count == 4u);

    make_token_matrix_dataset(test_case->dataset_kind, &dataset);
    CHECK(breakout_pufferlib_token_policy_train(&policy, &dataset, &config, &report) == TKM_OK);
    CHECK(strcmp(report.sequence_layout_name, test_case->layout_kind) == 0);
    CHECK(strcmp(report.observation_tokenizer_name, test_case->observation_tokenizer_kind) == 0);
    CHECK(strcmp(report.observation_selection_name, test_case->observation_selection_kind) == 0);
    CHECK(strcmp(report.sequence_model_name, test_case->model_kind) == 0);
    CHECK(strcmp(report.action_head_name, test_case->head_kind) == 0);
    CHECK(policy.model_kind == test_case->expected_model_kind);
    CHECK(policy.head_kind == test_case->expected_head_kind);
    CHECK(policy.selected_dims[0] < TKM_PUFFERLIB_BREAKOUT_OBS_DIM);
    if (test_case->expected_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS) {
        CHECK(policy.selected_dims[0] == 0u);
        CHECK(policy.selected_dims[1] == 1u);
        CHECK(policy.selected_dims[2] == 2u);
        CHECK(policy.selected_dims[3] == 3u);
    }
    if (test_case->expected_selection_kind == BREAKOUT_PUFFERLIB_TOKEN_SELECTION_MANUAL) {
        CHECK(policy.selected_dims[0] == test_case->expected_manual_first_dim);
    }
    CHECK(report.train_rows > 0u);
    CHECK(report.heldout_rows > 0u);

    if (test_case->dataset_kind == TOKEN_MATRIX_DATASET_RELATION) {
        obs[0] = 0.125f;
        obs[1] = 0.875f;
    } else {
        set_autoregressive_pattern_obs(obs, 2u);
    }
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT);
}

static void test_breakout_token_layer_permutation_matrix_accepts_supported_combinations(void) {
    static const BreakoutTokenMatrixCase cases[] = {
        {
            "token_ngram_categorical",
            "autoregressive_obs_action",
            "next_token",
            "selected_quantized",
            "first_dims",
            NULL,
            "token_ngram",
            "categorical",
            BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_NGRAM,
            BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL,
            BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS,
            TOKEN_MATRIX_DATASET_AUTOREGRESSIVE,
            0u
        },
        {
            "token_backoff_ngram_categorical",
            "autoregressive_obs_action",
            "next_token",
            "selected_quantized",
            "first_dims",
            NULL,
            "token_backoff_ngram",
            "categorical",
            BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_BACKOFF_NGRAM,
            BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL,
            BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS,
            TOKEN_MATRIX_DATASET_AUTOREGRESSIVE,
            0u
        },
        {
            "token_mlp_categorical",
            "autoregressive_obs_action",
            "next_token",
            "selected_quantized",
            "first_dims",
            NULL,
            "token_mlp_window",
            "categorical",
            BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW,
            BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL,
            BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS,
            TOKEN_MATRIX_DATASET_AUTOREGRESSIVE,
            0u
        },
        {
            "token_mlp_categorical_first_dims_repeat",
            "autoregressive_obs_action",
            "next_token",
            "selected_quantized",
            "first_dims",
            NULL,
            "token_mlp_window",
            "categorical",
            BREAKOUT_PUFFERLIB_TOKEN_MODEL_TOKEN_MLP_WINDOW,
            BREAKOUT_PUFFERLIB_TOKEN_HEAD_CATEGORICAL,
            BREAKOUT_PUFFERLIB_TOKEN_SELECTION_FIRST_DIMS,
            TOKEN_MATRIX_DATASET_AUTOREGRESSIVE,
            0u
        }
    };
    const uint32_t case_count = (uint32_t)(sizeof(cases) / sizeof(cases[0]));

    for (uint32_t i = 0u; i < case_count; i++) {
        check_token_matrix_case(&cases[i]);
    }
}

static void check_token_matrix_rejected(
    const char* layout_kind,
    const char* predict_kind,
    const char* observation_tokenizer_kind,
    const char* observation_selection_kind,
    const char* observation_indices,
    const char* model_kind,
    const char* head_kind
) {
    BreakoutTokenMatrixCase test_case = {
        "invalid",
        layout_kind,
        predict_kind,
        observation_tokenizer_kind,
        observation_selection_kind,
        observation_indices,
        model_kind,
        head_kind,
        BREAKOUT_PUFFERLIB_TOKEN_MODEL_UNSPECIFIED,
        BREAKOUT_PUFFERLIB_TOKEN_HEAD_UNSPECIFIED,
        BREAKOUT_PUFFERLIB_TOKEN_SELECTION_UNSPECIFIED,
        TOKEN_MATRIX_DATASET_AUTOREGRESSIVE,
        0u
    };
    TkmIni ini;
    BreakoutPufferlibTokenConfig config;

    CHECK(tkm_ini_parse(&ini, token_matrix_ini_text(&test_case)) == TKM_OK);
    CHECK(breakout_pufferlib_token_config_from_ini(&ini, &config) == TKM_ERR);
}

static void test_breakout_token_layer_permutation_matrix_rejects_unsupported_combinations(void) {
    check_token_matrix_rejected(
        "autoregressive_obs_action",
        "next_token",
        "selected_quantized",
        "first_dims",
        NULL,
        "token_ngram",
        "unknown_head");
    check_token_matrix_rejected(
        "autoregressive_obs_action",
        "next_token",
        "selected_quantized",
        "first_dims",
        NULL,
        "direct_action_context",
        "categorical");
    check_token_matrix_rejected(
        "obs_action_interleaved",
        "next_token",
        "selected_quantized",
        "first_dims",
        NULL,
        "token_mlp_window",
        "categorical");
    check_token_matrix_rejected(
        "autoregressive_obs_action",
        "action",
        "selected_quantized",
        "manual",
        NULL,
        "direct_action_context",
        "categorical");
    check_token_matrix_rejected(
        "autoregressive_obs_action",
        "action",
        "selected_quantized",
        "manual",
        "3,2,1\n",
        "direct_action_context",
        "categorical");
    check_token_matrix_rejected(
        "autoregressive_obs_action",
        "action",
        "selected_quantized",
        "first_dims",
        NULL,
        "token_ngram",
        "categorical");
    check_token_matrix_rejected(
        "obs_action_interleaved",
        "action",
        "selected_quantized",
        "first_dims",
        NULL,
        "mlp_window",
        "categorical");
}

static void test_breakout_render_visualization_parser_accepts_overlay_modes(void) {
    BreakoutPufferlibRenderVisualizationKind kind;

    CHECK(breakout_pufferlib_render_visualization_kind_from_string("none", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_NONE);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("topk", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_TOPK);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("token_stream", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_TOKEN_STREAM);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("continuous_mlp", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_CONTINUOUS_MLP);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("context_grid", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_CONTEXT_GRID);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("quantization", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_QUANTIZATION);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("pipeline", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_PIPELINE);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("transformer", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_TRANSFORMER);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("all", &kind) == TKM_OK);
    CHECK(kind == BREAKOUT_PUFFERLIB_RENDER_VIS_ALL);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("token_tape", &kind) == TKM_ERR);
    CHECK(breakout_pufferlib_render_visualization_kind_from_string("oracle", &kind) == TKM_ERR);
}

int main(void) {
    test_breakout_token_policy_trains_next_action_from_quantized_tokens();
    test_breakout_token_config_selects_autoregressive_next_token_backend();
    test_breakout_token_backoff_ngram_uses_suffix_before_prior();
    test_breakout_token_mlp_window_predicts_categoricals();
    test_breakout_tokenizer_first_dims_selects_prefix_dims();
    test_breakout_tokenizer_manual_selection_uses_ini_indices();
    test_breakout_token_mlp_window_uses_categorical_full_vocab_head();
    test_breakout_continuous_mlp_first_dims_predicts_actions();
    test_breakout_token_layer_permutation_matrix_accepts_supported_combinations();
    test_breakout_token_layer_permutation_matrix_rejects_unsupported_combinations();
    test_breakout_render_visualization_parser_accepts_overlay_modes();
    puts("breakout pufferlib policy tests passed");
    return 0;
}
