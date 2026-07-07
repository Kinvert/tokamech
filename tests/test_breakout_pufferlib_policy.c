#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/dataset/float_transition.h"
#include "projects/breakout/pufferlib_policy.h"

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static void check_close(float actual, float expected, float tolerance) {
    float diff = actual - expected;
    if (diff < 0.0f) {
        diff = -diff;
    }
    CHECK(diff <= tolerance);
}

static uint8_t action_for_value(float value) {
    if (value < -0.25f) {
        return 0;
    }
    if (value > 0.25f) {
        return 2;
    }
    return 1;
}

static void add_policy_row(
    TkmFloatTransitionDataset* dataset,
    int32_t episode,
    int32_t t,
    float value,
    float reward,
    uint8_t terminal
) {
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
    row->obs[0] = value;
    row->obs[1] = value * value;
    row->action = action_for_value(value);
    row->reward = reward;
    row->terminal = terminal;
}

static void make_policy_dataset(TkmFloatTransitionDataset* dataset) {
    const float values[3] = {-1.0f, 0.0f, 1.0f};

    tkm_float_transition_dataset_init(dataset);
    for (int32_t i = 0; i < 30; i++) {
        add_policy_row(dataset, 0, i, values[i % 3], 0.0f, i == 29 ? 1u : 0u);
    }
    for (int32_t i = 0; i < 60; i++) {
        add_policy_row(dataset, 1, i, values[i % 3], 1.0f, i == 59 ? 1u : 0u);
    }
}

static BreakoutPufferlibBcConfig test_config(void) {
    BreakoutPufferlibBcConfig config;
    config.hidden_dim = 8;
    config.epochs = 80;
    config.learning_rate = 0.05f;
    config.heldout_stride = 5;
    return config;
}

static void test_breakout_policy_trains_action_classifier_and_reports_accuracy(void) {
    static TkmFloatTransitionDataset dataset;
    BreakoutPufferlibPolicy policy;
    BreakoutPufferlibBcReport report;
    BreakoutPufferlibBcConfig config = test_config();
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99;

    make_policy_dataset(&dataset);
    CHECK(breakout_pufferlib_policy_train_bc(&policy, &dataset, &config, "all", &report) == TKM_OK);
    CHECK(report.source_rows == 90);
    CHECK(report.train_rows > 0);
    CHECK(report.heldout_rows > 0);
    CHECK(strcmp(report.filter_name, "all") == 0);
    CHECK(report.heldout_accuracy >= 0.80f);

    obs[0] = -1.0f;
    CHECK(breakout_pufferlib_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 0);
    obs[0] = 0.0f;
    CHECK(breakout_pufferlib_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 1);
    obs[0] = 1.0f;
    CHECK(breakout_pufferlib_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 2);
}

static void test_breakout_policy_reports_filter_metadata(void) {
    static TkmFloatTransitionDataset dataset;
    static TkmFloatTransitionDataset top;
    BreakoutPufferlibPolicy policy_all;
    BreakoutPufferlibPolicy policy_top;
    BreakoutPufferlibBcReport report_all;
    BreakoutPufferlibBcReport report_top;
    BreakoutPufferlibBcConfig config = test_config();

    make_policy_dataset(&dataset);
    CHECK(tkm_float_transition_filter_top_return(&dataset, &top, 1) == TKM_OK);
    CHECK(top.row_count == 60);

    CHECK(breakout_pufferlib_policy_train_bc(&policy_all, &dataset, &config, "all", &report_all) == TKM_OK);
    CHECK(breakout_pufferlib_policy_train_bc(&policy_top, &top, &config, "top_return", &report_top) == TKM_OK);
    CHECK(report_all.source_rows == 90);
    CHECK(report_top.source_rows == 60);
    CHECK(strcmp(report_all.filter_name, "all") == 0);
    CHECK(strcmp(report_top.filter_name, "top_return") == 0);
}

static void test_breakout_policy_saves_and_loads_config_and_params(void) {
    static TkmFloatTransitionDataset dataset;
    BreakoutPufferlibPolicy policy;
    BreakoutPufferlibPolicy loaded;
    BreakoutPufferlibBcReport report;
    BreakoutPufferlibBcConfig config = test_config();
    const char* config_path = "/tmp/tkm_breakout_pufferlib_policy.ini";
    const char* params_path = "/tmp/tkm_breakout_pufferlib_policy.params";
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action_before = 99;
    uint8_t action_after = 88;

    make_policy_dataset(&dataset);
    CHECK(breakout_pufferlib_policy_train_bc(&policy, &dataset, &config, "all", &report) == TKM_OK);
    CHECK(breakout_pufferlib_policy_save(&policy, config_path, params_path) == TKM_OK);
    CHECK(breakout_pufferlib_policy_load(&loaded, config_path, params_path) == TKM_OK);

    obs[0] = 1.0f;
    CHECK(breakout_pufferlib_policy_predict(&policy, obs, &action_before) == TKM_OK);
    CHECK(breakout_pufferlib_policy_predict(&loaded, obs, &action_after) == TKM_OK);
    CHECK(action_before == action_after);
    check_close(loaded.obs_dim, TKM_PUFFERLIB_BREAKOUT_OBS_DIM, 0.0f);
    CHECK(loaded.action_count == TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT);
    CHECK(loaded.hidden_dim == config.hidden_dim);
}

static void test_breakout_nearest_policy_predicts_from_closest_observation(void) {
    static TkmFloatTransitionDataset dataset;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99;

    make_policy_dataset(&dataset);

    obs[0] = -0.9f;
    obs[1] = obs[0] * obs[0];
    CHECK(breakout_pufferlib_nearest_predict(&dataset, obs, &action) == TKM_OK);
    CHECK(action == 0);

    obs[0] = 0.05f;
    obs[1] = obs[0] * obs[0];
    CHECK(breakout_pufferlib_nearest_predict(&dataset, obs, &action) == TKM_OK);
    CHECK(action == 1);

    obs[0] = 0.9f;
    obs[1] = obs[0] * obs[0];
    CHECK(breakout_pufferlib_nearest_predict(&dataset, obs, &action) == TKM_OK);
    CHECK(action == 2);
}

static void test_breakout_nearest_policy_can_search_row_range(void) {
    static TkmFloatTransitionDataset dataset;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99;
    uint32_t index = 99;
    float distance = -1.0f;

    tkm_float_transition_dataset_init(&dataset);
    add_policy_row(&dataset, 0, 0, -1.0f, 0.0f, 0u);
    add_policy_row(&dataset, 0, 1, 0.0f, 0.0f, 0u);
    add_policy_row(&dataset, 0, 2, 1.0f, 0.0f, 1u);

    obs[0] = -1.0f;
    obs[1] = 1.0f;
    CHECK(breakout_pufferlib_nearest_predict_range(&dataset, obs, 1, 2, &action, &index, &distance) == TKM_OK);
    CHECK(action == 1);
    CHECK(index == 1);
    check_close(distance, 2.0f, 0.0001f);

    CHECK(breakout_pufferlib_nearest_predict_range(&dataset, obs, 3, 1, &action, &index, &distance) == TKM_ERR);
}

static void test_breakout_sequence_cursor_follows_next_row_and_reanchors(void) {
    static TkmFloatTransitionDataset dataset;
    BreakoutPufferlibSequenceCursor cursor;
    BreakoutPufferlibSequenceMatch match = 0;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM] = {0.0f};
    uint8_t action = 99;
    uint32_t index = 99;
    float distance = -1.0f;

    tkm_float_transition_dataset_init(&dataset);
    add_policy_row(&dataset, 0, 0, -1.0f, 0.0f, 0u);
    add_policy_row(&dataset, 0, 1, 0.0f, 0.0f, 0u);
    add_policy_row(&dataset, 0, 2, 1.0f, 0.0f, 1u);
    add_policy_row(&dataset, 1, 0, -0.5f, 0.0f, 0u);

    CHECK(breakout_pufferlib_sequence_cursor_init(&cursor, 32u, 0.01f) == TKM_OK);

    obs[0] = -1.0f;
    obs[1] = 1.0f;
    CHECK(breakout_pufferlib_sequence_cursor_predict(
            &dataset, &cursor, obs, 0u, &action, &match, &index, &distance) == TKM_OK);
    CHECK(action == 0);
    CHECK(index == 0u);
    CHECK(cursor.cursor == 0u);
    CHECK(cursor.has_cursor == 1u);
    CHECK(match == BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_FULL_SEARCH);
    check_close(distance, 0.0f, 0.0001f);

    obs[0] = 0.0f;
    obs[1] = 0.0f;
    CHECK(breakout_pufferlib_sequence_cursor_predict(
            &dataset, &cursor, obs, 1u, &action, &match, &index, &distance) == TKM_OK);
    CHECK(action == 1);
    CHECK(index == 1u);
    CHECK(cursor.cursor == 1u);
    CHECK(match == BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_NEXT_ROW);
    check_close(distance, 0.0f, 0.0001f);

    obs[0] = -0.5f;
    obs[1] = 0.25f;
    CHECK(breakout_pufferlib_sequence_cursor_predict(
            &dataset, &cursor, obs, 2u, &action, &match, &index, &distance) == TKM_OK);
    CHECK(action == 0);
    CHECK(index == 3u);
    CHECK(cursor.cursor == 3u);
    CHECK(match == BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_FULL_SEARCH);
    check_close(distance, 0.0f, 0.0001f);

    breakout_pufferlib_sequence_cursor_reset(&cursor);
    CHECK(cursor.has_cursor == 0u);
}

static void set_intercept_obs(
    float* obs,
    float paddle_x,
    float ball_x,
    float ball_y,
    float ball_vx,
    float ball_vy
) {
    memset(obs, 0, sizeof(float) * TKM_PUFFERLIB_BREAKOUT_OBS_DIM);
    obs[0] = paddle_x;
    obs[1] = 312.0f / 330.0f;
    obs[2] = ball_x;
    obs[3] = ball_y;
    obs[4] = ball_vx;
    obs[5] = ball_vy;
    obs[6] = 0.2f;
    obs[8] = 1.0f;
    obs[9] = 1.0f;
}

static void test_breakout_intercept_policy_tracks_predicted_ball_x(void) {
    BreakoutPufferlibInterceptPolicy policy;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    uint8_t action = 99;

    CHECK(breakout_pufferlib_intercept_policy_init(&policy, 0.03f, 0.0f) == TKM_OK);

    set_intercept_obs(obs, 0.20f, 0.60f, 0.80f, 0.0f, 0.01f);
    CHECK(breakout_pufferlib_intercept_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 2);

    set_intercept_obs(obs, 0.70f, 0.20f, 0.80f, 0.0f, 0.01f);
    CHECK(breakout_pufferlib_intercept_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 1);

    set_intercept_obs(obs, 0.47f, 0.50f, 0.80f, 0.0f, 0.01f);
    CHECK(breakout_pufferlib_intercept_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 0);
}

static void test_breakout_intercept_policy_reflects_wall_intercept(void) {
    BreakoutPufferlibInterceptPolicy policy;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    uint8_t action = 99;

    CHECK(breakout_pufferlib_intercept_policy_init(&policy, 0.03f, 0.0f) == TKM_OK);
    set_intercept_obs(obs, 0.65f, 0.90f, 0.45f, 0.03f, 0.01f);

    CHECK(breakout_pufferlib_intercept_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 1);
}

static void add_intercept_row(
    TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibInterceptPolicy* teacher,
    float paddle_x,
    float ball_x,
    float ball_y,
    float ball_vx,
    float ball_vy,
    int32_t t
) {
    TkmFloatTransition* row;
    uint8_t action = 99;

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
    set_intercept_obs(row->obs, paddle_x, ball_x, ball_y, ball_vx, ball_vy);
    CHECK(breakout_pufferlib_intercept_policy_predict(teacher, row->obs, &action) == TKM_OK);
    row->action = action;
}

static void test_breakout_intercept_policy_trains_grid_from_rows(void) {
    static TkmFloatTransitionDataset dataset;
    BreakoutPufferlibInterceptPolicy teacher;
    BreakoutPufferlibInterceptPolicy trained;
    BreakoutPufferlibInterceptReport report;
    float obs[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    uint8_t action = 99;

    CHECK(breakout_pufferlib_intercept_policy_init(&teacher, 0.04f, 0.10f) == TKM_OK);
    tkm_float_transition_dataset_init(&dataset);
    for (int32_t i = 0; i < 90; i++) {
        float phase = (float)(i % 9);
        float paddle_x = 0.08f + 0.09f * (float)(i % 8);
        float ball_x = 0.05f + 0.10f * phase;
        float ball_y = 0.40f + 0.05f * (float)(i % 5);
        float ball_vx = ((i % 7) - 3) * 0.006f;
        float ball_vy = 0.010f + 0.002f * (float)(i % 3);
        add_intercept_row(&dataset, &teacher, paddle_x, ball_x, ball_y, ball_vx, ball_vy, i);
    }

    CHECK(breakout_pufferlib_intercept_policy_train_grid(&trained, &dataset, &report) == TKM_OK);
    CHECK(report.source_rows == 90u);
    CHECK(report.train_rows > 0u);
    CHECK(report.heldout_rows > 0u);
    CHECK(report.heldout_accuracy >= 0.80f);

    set_intercept_obs(obs, 0.20f, 0.60f, 0.80f, 0.0f, 0.01f);
    CHECK(breakout_pufferlib_intercept_policy_predict(&trained, obs, &action) == TKM_OK);
    CHECK(action == 2);
}

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
    CHECK(action == 1u);

    obs[0] = 0.875f;
    obs[1] = 0.125f;
    CHECK(breakout_pufferlib_token_policy_reset(&policy) == TKM_OK);
    CHECK(breakout_pufferlib_token_policy_predict(&policy, obs, &action) == TKM_OK);
    CHECK(action == 2u);
}

int main(void) {
    test_breakout_policy_trains_action_classifier_and_reports_accuracy();
    test_breakout_policy_reports_filter_metadata();
    test_breakout_policy_saves_and_loads_config_and_params();
    test_breakout_nearest_policy_predicts_from_closest_observation();
    test_breakout_nearest_policy_can_search_row_range();
    test_breakout_sequence_cursor_follows_next_row_and_reanchors();
    test_breakout_intercept_policy_tracks_predicted_ball_x();
    test_breakout_intercept_policy_reflects_wall_intercept();
    test_breakout_intercept_policy_trains_grid_from_rows();
    test_breakout_token_policy_trains_next_action_from_quantized_tokens();
    puts("breakout pufferlib policy tests passed");
    return 0;
}
