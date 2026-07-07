#include "projects/breakout/pufferlib_policy.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/config/ini.h"

#define BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN 64u

static int breakout_policy_ready(const BreakoutPufferlibPolicy* policy) {
    return policy &&
        policy->obs_dim == TKM_PUFFERLIB_BREAKOUT_OBS_DIM &&
        policy->action_count == TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT &&
        policy->hidden_dim > 0 &&
        policy->hidden_dim <= BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN;
}

static int breakout_policy_copy_filter_name(char* dst, const char* src) {
    uint32_t i = 0;

    if (!dst) {
        return TKM_ERR;
    }
    if (!src) {
        src = "unspecified";
    }

    while (src[i] != '\0') {
        if (i + 1u >= BREAKOUT_PUFFERLIB_POLICY_MAX_FILTER_NAME) {
            return TKM_ERR;
        }
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
    return TKM_OK;
}

int breakout_pufferlib_policy_init(BreakoutPufferlibPolicy* policy, uint32_t hidden_dim) {
    float w0[TKM_PUFFERLIB_BREAKOUT_OBS_DIM * BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN];
    float b0[BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN];
    float w1[BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    float b1[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];

    if (!policy || hidden_dim == 0 || hidden_dim > BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < TKM_PUFFERLIB_BREAKOUT_OBS_DIM * hidden_dim; i++) {
        w0[i] = 0.0f;
    }
    for (uint32_t h = 0; h < hidden_dim; h++) {
        b0[h] = 0.05f;
        if (h == 0u) {
            w0[h] = -1.0f;
            b0[h] = 0.25f;
        } else if (h == 1u) {
            b0[h] = 1.0f;
        } else if (h == 2u) {
            w0[h] = 1.0f;
            b0[h] = 0.25f;
        } else {
            uint32_t input_index = h % TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
            w0[input_index * hidden_dim + h] = 0.01f * (float)((h % 5u) + 1u);
        }
    }
    for (uint32_t h = 0; h < hidden_dim; h++) {
        for (uint32_t action = 0; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
            int pattern = (int)(((h + 1u) * (action + 3u)) % 7u);
            w1[h * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT + action] = 0.01f * (float)(pattern - 3);
        }
    }
    for (uint32_t action = 0; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        b1[action] = 0.0f;
    }

    policy->obs_dim = TKM_PUFFERLIB_BREAKOUT_OBS_DIM;
    policy->action_count = TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT;
    policy->hidden_dim = hidden_dim;

    return tkm_mlp_window_init(
        &policy->mlp,
        TKM_PUFFERLIB_BREAKOUT_OBS_DIM,
        hidden_dim,
        TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT,
        w0,
        b0,
        w1,
        b1
    );
}

int breakout_pufferlib_policy_predict(
    const BreakoutPufferlibPolicy* policy,
    const float* obs,
    uint8_t* out_action
) {
    float logits[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    uint8_t best = 0;

    if (!breakout_policy_ready(policy) || !obs || !out_action) {
        return TKM_ERR;
    }
    if (tkm_mlp_window_forward(&policy->mlp, obs, logits, TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint8_t action = 1; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        if (logits[action] > logits[best]) {
            best = action;
        }
    }

    *out_action = best;
    return TKM_OK;
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

static float breakout_obs_distance(const float* a, const float* b) {
    float distance = 0.0f;

    for (uint32_t i = 0; i < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; i++) {
        float diff = a[i] - b[i];
        distance += diff * diff;
    }
    return distance;
}

static float breakout_clampf(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static float breakout_reflect_x(float x, float max_x) {
    float period;

    if (max_x <= 0.0f) {
        return 0.0f;
    }
    period = 2.0f * max_x;
    while (x < 0.0f) {
        x += period;
    }
    while (x > period) {
        x -= period;
    }
    if (x > max_x) {
        x = period - x;
    }
    return x;
}

static int breakout_is_heldout(uint32_t row_index, uint32_t heldout_stride) {
    return heldout_stride > 0u && row_index % heldout_stride == 0u;
}

int breakout_pufferlib_policy_train_bc(
    BreakoutPufferlibPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibBcConfig* config,
    const char* filter_name,
    BreakoutPufferlibBcReport* report
) {
    uint32_t train_rows = 0;
    uint32_t heldout_rows = 0;
    uint32_t heldout_correct = 0;

    if (!policy ||
        !breakout_dataset_ready(dataset) ||
        !config ||
        config->hidden_dim == 0 ||
        config->hidden_dim > BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN ||
        config->epochs == 0 ||
        config->learning_rate <= 0.0f ||
        config->heldout_stride < 2u ||
        !report) {
        return TKM_ERR;
    }

    if (breakout_pufferlib_policy_init(policy, config->hidden_dim) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0; i < dataset->row_count; i++) {
        if (breakout_is_heldout(i, config->heldout_stride)) {
            heldout_rows++;
        } else {
            train_rows++;
        }
    }
    if (train_rows == 0 || heldout_rows == 0) {
        return TKM_ERR;
    }

    for (uint32_t epoch = 0; epoch < config->epochs; epoch++) {
        for (uint32_t i = 0; i < dataset->row_count; i++) {
            const TkmFloatTransition* row = &dataset->rows[i];
            if (breakout_is_heldout(i, config->heldout_stride)) {
                continue;
            }
            if (row->obs_dim != TKM_PUFFERLIB_BREAKOUT_OBS_DIM ||
                row->action >= TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT ||
                tkm_mlp_window_train_cross_entropy(&policy->mlp, row->obs, row->action, config->learning_rate) != TKM_OK) {
                return TKM_ERR;
            }
        }
    }

    for (uint32_t i = 0; i < dataset->row_count; i++) {
        const TkmFloatTransition* row = &dataset->rows[i];
        uint8_t action = 0;
        if (!breakout_is_heldout(i, config->heldout_stride)) {
            continue;
        }
        if (breakout_pufferlib_policy_predict(policy, row->obs, &action) != TKM_OK) {
            return TKM_ERR;
        }
        if (action == row->action) {
            heldout_correct++;
        }
    }

    report->source_rows = dataset->row_count;
    report->train_rows = train_rows;
    report->heldout_rows = heldout_rows;
    report->heldout_accuracy = (float)heldout_correct / (float)heldout_rows;
    return breakout_policy_copy_filter_name(report->filter_name, filter_name);
}

static int breakout_write_array(FILE* file, const char* name, const float* values, uint32_t count) {
    if (!file || !name || !values || count == 0) {
        return TKM_ERR;
    }

    if (fprintf(file, "%s %u\n", name, count) < 0) {
        return TKM_ERR;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (fprintf(file, "%.9g\n", values[i]) < 0) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

int breakout_pufferlib_policy_save(
    const BreakoutPufferlibPolicy* policy,
    const char* config_path,
    const char* params_path
) {
    FILE* config_file;
    FILE* params_file;
    uint32_t w0_count;
    uint32_t b0_count;
    uint32_t w1_count;
    uint32_t b1_count;

    if (!breakout_policy_ready(policy) || !config_path || !params_path) {
        return TKM_ERR;
    }

    config_file = fopen(config_path, "wb");
    if (!config_file) {
        return TKM_ERR;
    }
    if (fprintf(config_file,
            "[policy]\n"
            "env = pufferlib_breakout\n"
            "obs_dim = %u\n"
            "action_count = %u\n"
            "hidden_dim = %u\n",
            policy->obs_dim,
            policy->action_count,
            policy->hidden_dim
        ) < 0 ||
        fclose(config_file) != 0) {
        return TKM_ERR;
    }

    params_file = fopen(params_path, "wb");
    if (!params_file) {
        return TKM_ERR;
    }
    w0_count = policy->obs_dim * policy->hidden_dim;
    b0_count = policy->hidden_dim;
    w1_count = policy->hidden_dim * policy->action_count;
    b1_count = policy->action_count;
    if (fprintf(params_file, "breakout_pufferlib_policy_params_v1\n") < 0 ||
        breakout_write_array(params_file, "w0", policy->mlp.w0, w0_count) != TKM_OK ||
        breakout_write_array(params_file, "b0", policy->mlp.b0, b0_count) != TKM_OK ||
        breakout_write_array(params_file, "w1", policy->mlp.w1, w1_count) != TKM_OK ||
        breakout_write_array(params_file, "b1", policy->mlp.b1, b1_count) != TKM_OK ||
        fclose(params_file) != 0) {
        return TKM_ERR;
    }

    return TKM_OK;
}

static int breakout_read_array(FILE* file, const char* expected_name, float* values, uint32_t expected_count) {
    char name[64];
    unsigned int count = 0;

    if (!file || !expected_name || !values || expected_count == 0) {
        return TKM_ERR;
    }
    if (fscanf(file, "%63s %u", name, &count) != 2 ||
        strcmp(name, expected_name) != 0 ||
        count != expected_count) {
        return TKM_ERR;
    }
    for (uint32_t i = 0; i < expected_count; i++) {
        if (fscanf(file, "%f", &values[i]) != 1) {
            return TKM_ERR;
        }
    }

    return TKM_OK;
}

int breakout_pufferlib_policy_load(
    BreakoutPufferlibPolicy* policy,
    const char* config_path,
    const char* params_path
) {
    TkmIni ini;
    int32_t obs_dim = 0;
    int32_t action_count = 0;
    int32_t hidden_dim = 0;
    FILE* params_file;
    char magic[64];
    float w0[TKM_PUFFERLIB_BREAKOUT_OBS_DIM * BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN];
    float b0[BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN];
    float w1[BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    float b1[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];

    if (!policy || !config_path || !params_path) {
        return TKM_ERR;
    }

    if (tkm_ini_parse_file(&ini, config_path) != TKM_OK ||
        strcmp(tkm_ini_get(&ini, "policy", "env", ""), "pufferlib_breakout") != 0 ||
        tkm_ini_get_i32(&ini, "policy", "obs_dim", 0, &obs_dim) != TKM_OK ||
        tkm_ini_get_i32(&ini, "policy", "action_count", 0, &action_count) != TKM_OK ||
        tkm_ini_get_i32(&ini, "policy", "hidden_dim", 0, &hidden_dim) != TKM_OK ||
        obs_dim != (int32_t)TKM_PUFFERLIB_BREAKOUT_OBS_DIM ||
        action_count != (int32_t)TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT ||
        hidden_dim <= 0 ||
        hidden_dim > (int32_t)BREAKOUT_PUFFERLIB_POLICY_MAX_HIDDEN) {
        return TKM_ERR;
    }

    params_file = fopen(params_path, "rb");
    if (!params_file) {
        return TKM_ERR;
    }
    if (fscanf(params_file, "%63s", magic) != 1 ||
        strcmp(magic, "breakout_pufferlib_policy_params_v1") != 0 ||
        breakout_read_array(params_file, "w0", w0, (uint32_t)obs_dim * (uint32_t)hidden_dim) != TKM_OK ||
        breakout_read_array(params_file, "b0", b0, (uint32_t)hidden_dim) != TKM_OK ||
        breakout_read_array(params_file, "w1", w1, (uint32_t)hidden_dim * (uint32_t)action_count) != TKM_OK ||
        breakout_read_array(params_file, "b1", b1, (uint32_t)action_count) != TKM_OK ||
        fclose(params_file) != 0) {
        return TKM_ERR;
    }

    policy->obs_dim = (uint32_t)obs_dim;
    policy->action_count = (uint32_t)action_count;
    policy->hidden_dim = (uint32_t)hidden_dim;
    return tkm_mlp_window_init(
        &policy->mlp,
        policy->obs_dim,
        policy->hidden_dim,
        policy->action_count,
        w0,
        b0,
        w1,
        b1
    );
}

int breakout_pufferlib_nearest_predict(
    const TkmFloatTransitionDataset* dataset,
    const float* obs,
    uint8_t* out_action
) {
    return breakout_pufferlib_nearest_predict_range(
        dataset, obs, 0u, dataset ? dataset->row_count : 0u, out_action, NULL, NULL);
}

int breakout_pufferlib_nearest_predict_range(
    const TkmFloatTransitionDataset* dataset,
    const float* obs,
    uint32_t start_index,
    uint32_t row_count,
    uint8_t* out_action,
    uint32_t* out_index,
    float* out_distance
) {
    uint32_t best_index = 0;
    float best_distance = 0.0f;
    uint32_t end_index;

    if (!breakout_dataset_ready(dataset) || !obs || !out_action ||
        start_index >= dataset->row_count || row_count == 0u) {
        return TKM_ERR;
    }
    if (row_count > dataset->row_count - start_index) {
        return TKM_ERR;
    }
    end_index = start_index + row_count;

    for (uint32_t row_index = start_index; row_index < end_index; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        float distance = 0.0f;

        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }

        distance = breakout_obs_distance(obs, row->obs);

        if (row_index == start_index || distance < best_distance) {
            best_distance = distance;
            best_index = row_index;
        }
    }

    *out_action = dataset->rows[best_index].action;
    if (out_index) {
        *out_index = best_index;
    }
    if (out_distance) {
        *out_distance = best_distance;
    }
    return TKM_OK;
}

int breakout_pufferlib_sequence_cursor_init(
    BreakoutPufferlibSequenceCursor* cursor,
    uint32_t reanchor_interval,
    float max_sequence_distance
) {
    if (!cursor || max_sequence_distance < 0.0f) {
        return TKM_ERR;
    }

    cursor->cursor = 0u;
    cursor->has_cursor = 0u;
    cursor->reanchor_interval = reanchor_interval;
    cursor->max_sequence_distance = max_sequence_distance;
    return TKM_OK;
}

void breakout_pufferlib_sequence_cursor_reset(BreakoutPufferlibSequenceCursor* cursor) {
    if (!cursor) {
        return;
    }
    cursor->cursor = 0u;
    cursor->has_cursor = 0u;
}

int breakout_pufferlib_sequence_cursor_predict(
    const TkmFloatTransitionDataset* dataset,
    BreakoutPufferlibSequenceCursor* cursor,
    const float* obs,
    uint32_t step,
    uint8_t* out_action,
    BreakoutPufferlibSequenceMatch* out_match,
    uint32_t* out_index,
    float* out_distance
) {
    uint8_t use_full;

    if (!breakout_dataset_ready(dataset) || !cursor || !obs || !out_action) {
        return TKM_ERR;
    }

    use_full = !cursor->has_cursor ||
        cursor->cursor + 1u >= dataset->row_count ||
        (cursor->reanchor_interval > 0u && step % cursor->reanchor_interval == 0u);

    if (!use_full) {
        const TkmFloatTransition* current = &dataset->rows[cursor->cursor];
        const TkmFloatTransition* next = &dataset->rows[cursor->cursor + 1u];
        float distance;

        if (!breakout_transition_row_ready(current) || !breakout_transition_row_ready(next)) {
            return TKM_ERR;
        }

        distance = breakout_obs_distance(obs, next->obs);
        if (next->env_index == current->env_index &&
            next->episode == current->episode &&
            distance <= cursor->max_sequence_distance) {
            cursor->cursor++;
            *out_action = next->action;
            if (out_match) {
                *out_match = BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_NEXT_ROW;
            }
            if (out_index) {
                *out_index = cursor->cursor;
            }
            if (out_distance) {
                *out_distance = distance;
            }
            return TKM_OK;
        }
    }

    if (breakout_pufferlib_nearest_predict_range(
            dataset, obs, 0u, dataset->row_count, out_action, &cursor->cursor, out_distance) != TKM_OK) {
        return TKM_ERR;
    }
    cursor->has_cursor = 1u;
    if (out_match) {
        *out_match = BREAKOUT_PUFFERLIB_SEQUENCE_MATCH_FULL_SEARCH;
    }
    if (out_index) {
        *out_index = cursor->cursor;
    }
    return TKM_OK;
}

int breakout_pufferlib_intercept_policy_init(
    BreakoutPufferlibInterceptPolicy* policy,
    float deadzone,
    float aim_offset
) {
    if (!policy || deadzone < 0.0f || aim_offset < -0.45f || aim_offset > 0.45f) {
        return TKM_ERR;
    }

    policy->deadzone = deadzone;
    policy->aim_offset = aim_offset;
    return TKM_OK;
}

int breakout_pufferlib_intercept_policy_predict(
    const BreakoutPufferlibInterceptPolicy* policy,
    const float* obs,
    uint8_t* out_action
) {
    const float width = 576.0f;
    const float height = 330.0f;
    const float ball_width = 32.0f / width;
    const float ball_height = 32.0f / height;
    const float full_paddle_width = 62.0f / width;
    float paddle_width;
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float vx;
    float vy;
    float target_ball_left;
    float target_center;
    float target_paddle_x;
    float max_ball_x;
    float max_paddle_x;
    float diff;

    if (!policy || !obs || !out_action) {
        return TKM_ERR;
    }

    paddle_x = obs[0];
    paddle_y = obs[1];
    ball_x = obs[2];
    ball_y = obs[3];
    vx = obs[4] * 512.0f / width;
    vy = obs[5] * 512.0f / height;
    paddle_width = obs[9] > 0.0f ? obs[9] * full_paddle_width : full_paddle_width;
    max_ball_x = 1.0f - ball_width;
    max_paddle_x = 1.0f - paddle_width;
    target_ball_left = ball_x;

    if (vy > 0.000001f) {
        float frames_to_paddle = (paddle_y - (ball_y + ball_height)) / vy;
        if (frames_to_paddle > 0.0f) {
            target_ball_left = breakout_reflect_x(ball_x + vx * frames_to_paddle, max_ball_x);
        }
    }

    target_center = target_ball_left + 0.5f * ball_width;
    target_paddle_x = target_center - (0.5f + policy->aim_offset) * paddle_width;
    target_paddle_x = breakout_clampf(target_paddle_x, 0.0f, max_paddle_x);
    diff = target_paddle_x - paddle_x;

    if (fabsf(diff) <= policy->deadzone) {
        *out_action = 0u;
    } else if (diff < 0.0f) {
        *out_action = 1u;
    } else {
        *out_action = 2u;
    }
    return TKM_OK;
}

int breakout_pufferlib_intercept_policy_train_grid(
    BreakoutPufferlibInterceptPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    BreakoutPufferlibInterceptReport* report
) {
    BreakoutPufferlibInterceptPolicy candidate;
    BreakoutPufferlibInterceptPolicy best;
    uint32_t best_correct = 0u;
    uint32_t best_total = 0u;
    uint32_t train_rows = 0u;
    uint32_t heldout_rows = 0u;
    uint32_t heldout_correct = 0u;
    const float deadzones[] = {0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.08f};
    const float aim_offsets[] = {-0.25f, -0.20f, -0.15f, -0.10f, -0.05f, 0.0f,
        0.05f, 0.10f, 0.15f, 0.20f, 0.25f};

    if (!policy || !breakout_dataset_ready(dataset) || !report) {
        return TKM_ERR;
    }

    if (breakout_pufferlib_intercept_policy_init(&best, 0.03f, 0.0f) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t deadzone_index = 0u;
         deadzone_index < sizeof(deadzones) / sizeof(deadzones[0]);
         deadzone_index++) {
        for (uint32_t aim_index = 0u;
             aim_index < sizeof(aim_offsets) / sizeof(aim_offsets[0]);
             aim_index++) {
            uint32_t correct = 0u;
            uint32_t total = 0u;

            if (breakout_pufferlib_intercept_policy_init(
                    &candidate, deadzones[deadzone_index], aim_offsets[aim_index]) != TKM_OK) {
                return TKM_ERR;
            }
            for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
                const TkmFloatTransition* row = &dataset->rows[row_index];
                uint8_t action = 0u;

                if (!breakout_transition_row_ready(row)) {
                    return TKM_ERR;
                }
                if (breakout_is_heldout(row_index, 5u)) {
                    continue;
                }
                if (breakout_pufferlib_intercept_policy_predict(&candidate, row->obs, &action) != TKM_OK) {
                    return TKM_ERR;
                }
                if (action == row->action) {
                    correct++;
                }
                total++;
            }
            if (total == 0u) {
                return TKM_ERR;
            }
            if (correct > best_correct || best_total == 0u) {
                best = candidate;
                best_correct = correct;
                best_total = total;
            }
        }
    }

    for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        uint8_t action = 0u;

        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }
        if (breakout_is_heldout(row_index, 5u)) {
            if (breakout_pufferlib_intercept_policy_predict(&best, row->obs, &action) != TKM_OK) {
                return TKM_ERR;
            }
            if (action == row->action) {
                heldout_correct++;
            }
            heldout_rows++;
        } else {
            train_rows++;
        }
    }
    if (train_rows == 0u || heldout_rows == 0u) {
        return TKM_ERR;
    }

    *policy = best;
    report->source_rows = dataset->row_count;
    report->train_rows = train_rows;
    report->heldout_rows = heldout_rows;
    report->heldout_accuracy = (float)heldout_correct / (float)heldout_rows;
    report->deadzone = best.deadzone;
    report->aim_offset = best.aim_offset;
    return TKM_OK;
}

static uint32_t breakout_token_hash_mix(uint32_t hash, uint32_t value) {
    hash ^= value + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
    return hash;
}

static uint32_t breakout_token_feature_hash(
    uint32_t type,
    uint32_t a,
    uint32_t b,
    uint32_t c,
    uint32_t d
) {
    uint32_t hash = 2166136261u;

    hash = breakout_token_hash_mix(hash, type);
    hash = breakout_token_hash_mix(hash, a);
    hash = breakout_token_hash_mix(hash, b);
    hash = breakout_token_hash_mix(hash, c);
    hash = breakout_token_hash_mix(hash, d);
    return hash & (BREAKOUT_PUFFERLIB_TOKEN_FEATURES - 1u);
}

static uint32_t breakout_token_weight_offset(uint32_t feature, uint32_t action) {
    return feature * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT + action;
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
    return policy->selected_dim_count * (BREAKOUT_PUFFERLIB_TOKEN_HISTORY + 1u);
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

static float breakout_token_normalized_obs_value(
    const BreakoutPufferlibTokenPolicy* policy,
    uint32_t dim,
    float value
) {
    float min_value;
    float max_value;
    float span;
    float scaled;

    if (!policy || dim >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
        return 0.0f;
    }
    min_value = policy->obs_min[dim];
    max_value = policy->obs_max[dim];
    span = max_value - min_value;
    if (span <= 0.000001f) {
        return 0.0f;
    }
    scaled = (value - min_value) / span;
    if (scaled < 0.0f) {
        scaled = 0.0f;
    }
    if (scaled > 1.0f) {
        scaled = 1.0f;
    }
    return 2.0f * scaled - 1.0f;
}

static int breakout_token_make_selected_values(
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
        out_values[i] = breakout_token_normalized_obs_value(policy, dim, obs[dim]);
    }
    return TKM_OK;
}

static int breakout_token_build_mlp_input(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS],
    uint32_t prev_obs_count,
    float* input,
    uint32_t input_capacity
) {
    float current_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    uint32_t input_index = 0u;

    if (!policy || !obs || !prev_obs_values || !input ||
        policy->bin_count < 2u ||
        policy->selected_dim_count == 0u ||
        policy->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS ||
        input_capacity < breakout_token_mlp_input_dim(policy)) {
        return TKM_ERR;
    }
    if (breakout_token_make_selected_values(policy, obs, current_values) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
        input[input_index++] = current_values[i];
    }

    for (uint32_t lag = 0u; lag < BREAKOUT_PUFFERLIB_TOKEN_HISTORY; lag++) {
        uint8_t has_lag = lag < prev_obs_count ? 1u : 0u;
        uint32_t history_index = has_lag ? prev_obs_count - 1u - lag : 0u;

        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            input[input_index++] = has_lag ? prev_obs_values[history_index][i] : 0.0f;
        }
    }

    return TKM_OK;
}

static int breakout_token_mlp_init(BreakoutPufferlibTokenPolicy* policy) {
    float w0[TKM_MLP_WINDOW_MAX_INPUT * BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN];
    float b0[BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN];
    float w1[BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    float b1[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    uint32_t input_dim;

    if (!policy) {
        return TKM_ERR;
    }
    input_dim = breakout_token_mlp_input_dim(policy);
    if (input_dim == 0u || input_dim > TKM_MLP_WINDOW_MAX_INPUT) {
        return TKM_ERR;
    }

    for (uint32_t i = 0u; i < input_dim; i++) {
        for (uint32_t h = 0u; h < BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN; h++) {
            int pattern = (int)(((i + 3u) * (h + 5u)) % 17u) - 8;
            w0[i * BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN + h] = 0.005f * (float)pattern;
        }
    }
    for (uint32_t h = 0u; h < BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN; h++) {
        b0[h] = 0.02f;
        for (uint32_t action = 0u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
            int pattern = (int)(((h + 11u) * (action + 7u)) % 13u) - 6;
            w1[h * TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT + action] = 0.005f * (float)pattern;
        }
    }
    for (uint32_t action = 0u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        b1[action] = 0.0f;
    }

    if (tkm_mlp_window_init(
            &policy->mlp,
            input_dim,
            BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN,
            TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT,
            w0,
            b0,
            w1,
            b1) != TKM_OK) {
        return TKM_ERR;
    }
    policy->use_mlp = 1u;
    policy->token_mlp_input_dim = input_dim;
    policy->token_mlp_hidden_dim = BREAKOUT_PUFFERLIB_TOKEN_MLP_HIDDEN;
    return TKM_OK;
}

static void breakout_token_history_reset(uint8_t* prev_actions, uint32_t* prev_action_count) {
    if (!prev_actions || !prev_action_count) {
        return;
    }
    for (uint32_t i = 0u; i < BREAKOUT_PUFFERLIB_TOKEN_HISTORY; i++) {
        prev_actions[i] = 0u;
    }
    *prev_action_count = 0u;
}

static void breakout_token_history_push(uint8_t* prev_actions, uint32_t* prev_action_count, uint8_t action) {
    uint32_t count;

    if (!prev_actions || !prev_action_count) {
        return;
    }
    count = *prev_action_count;
    if (count < BREAKOUT_PUFFERLIB_TOKEN_HISTORY) {
        prev_actions[count] = action;
        *prev_action_count = count + 1u;
        return;
    }
    for (uint32_t i = 1u; i < BREAKOUT_PUFFERLIB_TOKEN_HISTORY; i++) {
        prev_actions[i - 1u] = prev_actions[i];
    }
    prev_actions[BREAKOUT_PUFFERLIB_TOKEN_HISTORY - 1u] = action;
}

static void breakout_token_obs_history_reset(
    float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS],
    uint32_t* prev_obs_count
) {
    if (!prev_obs_values || !prev_obs_count) {
        return;
    }
    memset(
        prev_obs_values,
        0,
        BREAKOUT_PUFFERLIB_TOKEN_HISTORY * BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS * sizeof(float));
    *prev_obs_count = 0u;
}

static void breakout_token_obs_history_push(
    float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS],
    uint32_t* prev_obs_count,
    const float* values,
    uint32_t value_count
) {
    uint32_t count;

    if (!prev_obs_values || !prev_obs_count || !values ||
        value_count == 0u ||
        value_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return;
    }
    count = *prev_obs_count;
    if (count < BREAKOUT_PUFFERLIB_TOKEN_HISTORY) {
        memcpy(prev_obs_values[count], values, value_count * sizeof(float));
        *prev_obs_count = count + 1u;
        return;
    }
    for (uint32_t i = 1u; i < BREAKOUT_PUFFERLIB_TOKEN_HISTORY; i++) {
        memcpy(prev_obs_values[i - 1u], prev_obs_values[i], value_count * sizeof(float));
    }
    memcpy(prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY - 1u], values, value_count * sizeof(float));
}

static int breakout_token_build_features(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    const uint8_t* prev_actions,
    uint32_t prev_action_count,
    uint32_t* out_features,
    uint32_t feature_capacity,
    uint32_t* out_feature_count
) {
    uint8_t tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    uint32_t count = 0u;

    if (!policy || !obs || !prev_actions || !out_features || !out_feature_count ||
        policy->bin_count == 0u ||
        policy->selected_dim_count == 0u ||
        policy->selected_dim_count > BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS) {
        return TKM_ERR;
    }

    if (count >= feature_capacity) {
        return TKM_ERR;
    }
    out_features[count++] = breakout_token_feature_hash(0u, 0u, 0u, 0u, 0u);

    for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
        uint32_t dim = policy->selected_dims[i];
        if (dim >= TKM_PUFFERLIB_BREAKOUT_OBS_DIM) {
            return TKM_ERR;
        }
        tokens[i] = breakout_token_quantize(policy, dim, obs[dim]);
        if (count >= feature_capacity) {
            return TKM_ERR;
        }
        out_features[count++] = breakout_token_feature_hash(1u, dim, tokens[i], 0u, 0u);
    }

    if (policy->use_pair_features) {
        for (uint32_t i = 0u; i < policy->selected_dim_count; i++) {
            for (uint32_t j = i + 1u; j < policy->selected_dim_count; j++) {
                if (count >= feature_capacity) {
                    return TKM_ERR;
                }
                out_features[count++] = breakout_token_feature_hash(
                    2u,
                    policy->selected_dims[i],
                    tokens[i],
                    policy->selected_dims[j],
                    tokens[j]);
            }
        }
    }

    for (uint32_t lag = 0u; lag < prev_action_count && lag < BREAKOUT_PUFFERLIB_TOKEN_HISTORY; lag++) {
        uint32_t index = prev_action_count - 1u - lag;
        if (count >= feature_capacity) {
            return TKM_ERR;
        }
        out_features[count++] = breakout_token_feature_hash(3u, lag, prev_actions[index], 0u, 0u);
    }

    *out_feature_count = count;
    return TKM_OK;
}

static int breakout_token_predict_with_history(
    const BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    const uint8_t* prev_actions,
    uint32_t prev_action_count,
    float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS],
    uint32_t prev_obs_count,
    uint8_t* out_action
) {
    uint32_t features[512];
    uint32_t feature_count = 0u;
    float scores[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT] = {0.0f, 0.0f, 0.0f};
    uint8_t best = 0u;

    if (!policy || !obs || !prev_actions || !out_action) {
        return TKM_ERR;
    }
    if (policy->use_mlp) {
        float input[TKM_MLP_WINDOW_MAX_INPUT];
        float logits[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];

        if (breakout_token_build_mlp_input(
                policy,
                obs,
                prev_obs_values,
                prev_obs_count,
                input,
                TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
            tkm_mlp_window_forward(
                &policy->mlp,
                input,
                logits,
                TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT) != TKM_OK) {
            return TKM_ERR;
        }
        for (uint8_t action = 1u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
            if (logits[action] > logits[best]) {
                best = action;
            }
        }
        *out_action = best;
        return TKM_OK;
    }
    if (breakout_token_build_features(
            policy,
            obs,
            prev_actions,
            prev_action_count,
            features,
            sizeof(features) / sizeof(features[0]),
            &feature_count) != TKM_OK) {
        return TKM_ERR;
    }

    for (uint32_t i = 0u; i < feature_count; i++) {
        uint32_t feature = features[i];
        for (uint8_t action = 0u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
            scores[action] += policy->weights[breakout_token_weight_offset(feature, action)];
        }
    }
    for (uint8_t action = 1u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        if (scores[action] > scores[best]) {
            best = action;
        }
    }
    *out_action = best;
    return TKM_OK;
}

static int breakout_token_select_dims(
    BreakoutPufferlibTokenPolicy* policy,
    const float* means,
    const float* mean_squares,
    float action_sums[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT][TKM_PUFFERLIB_BREAKOUT_OBS_DIM],
    const uint32_t* action_counts
) {
    uint8_t selected[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];

    if (!policy || !means || !mean_squares || !action_sums || !action_counts) {
        return TKM_ERR;
    }
    for (uint32_t i = 0u; i < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; i++) {
        selected[i] = 0u;
    }

    for (uint32_t pick = 0u; pick < policy->selected_dim_count; pick++) {
        uint32_t best_dim = 0u;
        float best_score = -1.0f;
        for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
            float variance = mean_squares[dim] - means[dim] * means[dim];
            float score = 0.0f;
            if (selected[dim]) {
                continue;
            }
            if (variance > 0.000001f) {
                for (uint32_t action = 0u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
                    if (action_counts[action] > 0u) {
                        float action_mean = action_sums[action][dim] / (float)action_counts[action];
                        float diff = action_mean - means[dim];
                        score += (float)action_counts[action] * diff * diff;
                    }
                }
                score /= variance;
            }
            if (score <= 0.0f) {
                score = variance;
            }
            if (score > best_score) {
                best_score = score;
                best_dim = dim;
            }
        }
        selected[best_dim] = 1u;
        policy->selected_dims[pick] = best_dim;
    }
    return TKM_OK;
}

static int breakout_token_policy_ready(const BreakoutPufferlibTokenPolicy* policy) {
    return policy &&
        policy->bin_count > 0u &&
        policy->bin_count <= BREAKOUT_PUFFERLIB_TOKEN_MAX_BINS &&
        policy->selected_dim_count > 0u &&
        policy->selected_dim_count <= BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS;
}

int breakout_pufferlib_token_policy_train(
    BreakoutPufferlibTokenPolicy* policy,
    const TkmFloatTransitionDataset* dataset,
    const BreakoutPufferlibTokenConfig* config,
    BreakoutPufferlibTokenReport* report
) {
    float sums[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    float sum_squares[TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    float action_sums[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT][TKM_PUFFERLIB_BREAKOUT_OBS_DIM];
    uint32_t action_counts[TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT];
    float mlp_learning_rate;
    uint32_t train_rows = 0u;
    uint32_t heldout_rows = 0u;
    uint32_t heldout_correct = 0u;

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
        config->heldout_stride < 2u) {
        return TKM_ERR;
    }

    memset(policy, 0, sizeof(*policy));
    policy->bin_count = config->bin_count;
    policy->selected_dim_count = config->selected_dim_count;
    policy->use_pair_features = config->use_pair_features ? 1u : 0u;

    for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
        policy->obs_min[dim] = dataset->rows[0].obs[dim];
        policy->obs_max[dim] = dataset->rows[0].obs[dim];
        sums[dim] = 0.0f;
        sum_squares[dim] = 0.0f;
    }
    for (uint32_t action = 0u; action < TKM_PUFFERLIB_BREAKOUT_ACTION_COUNT; action++) {
        action_counts[action] = 0u;
        for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
            action_sums[action][dim] = 0.0f;
        }
    }

    for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
        const TkmFloatTransition* row = &dataset->rows[row_index];
        if (!breakout_transition_row_ready(row)) {
            return TKM_ERR;
        }
        action_counts[row->action]++;
        for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
            float value = row->obs[dim];
            if (value < policy->obs_min[dim]) {
                policy->obs_min[dim] = value;
            }
            if (value > policy->obs_max[dim]) {
                policy->obs_max[dim] = value;
            }
            sums[dim] += value;
            sum_squares[dim] += value * value;
            action_sums[row->action][dim] += value;
        }
    }
    for (uint32_t dim = 0u; dim < TKM_PUFFERLIB_BREAKOUT_OBS_DIM; dim++) {
        sums[dim] /= (float)dataset->row_count;
        sum_squares[dim] /= (float)dataset->row_count;
    }
    if (breakout_token_select_dims(policy, sums, sum_squares, action_sums, action_counts) != TKM_OK) {
        return TKM_ERR;
    }
    if (breakout_token_mlp_init(policy) != TKM_OK) {
        return TKM_ERR;
    }
    mlp_learning_rate = config->learning_rate * 0.05f;
    if (mlp_learning_rate > 0.02f) {
        mlp_learning_rate = 0.02f;
    }

    for (uint32_t epoch = 0u; epoch < config->epochs; epoch++) {
        uint8_t prev_actions[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
        float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
        uint32_t prev_action_count = 0u;
        uint32_t prev_obs_count = 0u;
        int32_t prev_env = -1;
        int32_t prev_episode = -1;

        breakout_token_history_reset(prev_actions, &prev_action_count);
        breakout_token_obs_history_reset(prev_obs_values, &prev_obs_count);
        for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
            const TkmFloatTransition* row = &dataset->rows[row_index];
            uint8_t current_obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
            float current_obs_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
            uint8_t predicted = 0u;
            uint32_t features[512];
            uint32_t feature_count = 0u;

            if (row->env_index != prev_env || row->episode != prev_episode) {
                breakout_token_history_reset(prev_actions, &prev_action_count);
                breakout_token_obs_history_reset(prev_obs_values, &prev_obs_count);
                prev_env = row->env_index;
                prev_episode = row->episode;
            }
            if (breakout_token_make_selected_tokens(policy, row->obs, current_obs_tokens) != TKM_OK ||
                breakout_token_make_selected_values(policy, row->obs, current_obs_values) != TKM_OK) {
                return TKM_ERR;
            }

            if (!breakout_is_heldout(row_index, config->heldout_stride)) {
                if (breakout_token_predict_with_history(
                        policy,
                        row->obs,
                        prev_actions,
                        prev_action_count,
                        prev_obs_values,
                        prev_obs_count,
                        &predicted) != TKM_OK ||
                    breakout_token_build_features(
                        policy,
                        row->obs,
                        prev_actions,
                        prev_action_count,
                        features,
                        sizeof(features) / sizeof(features[0]),
                        &feature_count) != TKM_OK) {
                    return TKM_ERR;
                }
                if (predicted != row->action) {
                    for (uint32_t i = 0u; i < feature_count; i++) {
                        uint32_t feature = features[i];
                        policy->weights[breakout_token_weight_offset(feature, row->action)] += config->learning_rate;
                        policy->weights[breakout_token_weight_offset(feature, predicted)] -= config->learning_rate;
                    }
                }
                {
                    float token_input[TKM_MLP_WINDOW_MAX_INPUT];
                    if (breakout_token_build_mlp_input(
                            policy,
                            row->obs,
                            prev_obs_values,
                            prev_obs_count,
                            token_input,
                            TKM_MLP_WINDOW_MAX_INPUT) != TKM_OK ||
                        tkm_mlp_window_train_cross_entropy(
                            &policy->mlp,
                            token_input,
                            row->action,
                            mlp_learning_rate) != TKM_OK) {
                        return TKM_ERR;
                    }
                }
            }
            breakout_token_obs_history_push(
                prev_obs_values,
                &prev_obs_count,
                current_obs_values,
                policy->selected_dim_count);
            breakout_token_history_push(prev_actions, &prev_action_count, row->action);
        }
    }

    {
        uint8_t prev_actions[BREAKOUT_PUFFERLIB_TOKEN_HISTORY];
        float prev_obs_values[BREAKOUT_PUFFERLIB_TOKEN_HISTORY][BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
        uint32_t prev_action_count = 0u;
        uint32_t prev_obs_count = 0u;
        int32_t prev_env = -1;
        int32_t prev_episode = -1;

        breakout_token_history_reset(prev_actions, &prev_action_count);
        breakout_token_obs_history_reset(prev_obs_values, &prev_obs_count);
        for (uint32_t row_index = 0u; row_index < dataset->row_count; row_index++) {
            const TkmFloatTransition* row = &dataset->rows[row_index];
            uint8_t current_obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
            float current_obs_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
            uint8_t predicted = 0u;

            if (row->env_index != prev_env || row->episode != prev_episode) {
                breakout_token_history_reset(prev_actions, &prev_action_count);
                breakout_token_obs_history_reset(prev_obs_values, &prev_obs_count);
                prev_env = row->env_index;
                prev_episode = row->episode;
            }
            if (breakout_token_make_selected_tokens(policy, row->obs, current_obs_tokens) != TKM_OK ||
                breakout_token_make_selected_values(policy, row->obs, current_obs_values) != TKM_OK) {
                return TKM_ERR;
            }

            if (breakout_is_heldout(row_index, config->heldout_stride)) {
                if (breakout_token_predict_with_history(
                        policy,
                        row->obs,
                        prev_actions,
                        prev_action_count,
                        prev_obs_values,
                        prev_obs_count,
                        &predicted) != TKM_OK) {
                    return TKM_ERR;
                }
                if (predicted == row->action) {
                    heldout_correct++;
                }
                heldout_rows++;
            } else {
                train_rows++;
            }
            breakout_token_obs_history_push(
                prev_obs_values,
                &prev_obs_count,
                current_obs_values,
                policy->selected_dim_count);
            breakout_token_history_push(prev_actions, &prev_action_count, row->action);
        }
    }

    if (train_rows == 0u || heldout_rows == 0u) {
        return TKM_ERR;
    }

    breakout_pufferlib_token_policy_reset(policy);
    report->source_rows = dataset->row_count;
    report->train_rows = train_rows;
    report->heldout_rows = heldout_rows;
    report->heldout_accuracy = (float)heldout_correct / (float)heldout_rows;
    report->bin_count = policy->bin_count;
    report->selected_dim_count = policy->selected_dim_count;
    return TKM_OK;
}

int breakout_pufferlib_token_policy_reset(BreakoutPufferlibTokenPolicy* policy) {
    if (!breakout_token_policy_ready(policy)) {
        return TKM_ERR;
    }
    breakout_token_history_reset(policy->prev_actions, &policy->prev_action_count);
    breakout_token_obs_history_reset(policy->prev_obs_values, &policy->prev_obs_count);
    return TKM_OK;
}

int breakout_pufferlib_token_policy_predict(
    BreakoutPufferlibTokenPolicy* policy,
    const float* obs,
    uint8_t* out_action
) {
    uint8_t current_obs_tokens[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];
    float current_obs_values[BREAKOUT_PUFFERLIB_TOKEN_MAX_SELECTED_DIMS];

    if (!breakout_token_policy_ready(policy) || !obs || !out_action) {
        return TKM_ERR;
    }
    if (breakout_token_make_selected_tokens(policy, obs, current_obs_tokens) != TKM_OK ||
        breakout_token_make_selected_values(policy, obs, current_obs_values) != TKM_OK) {
        return TKM_ERR;
    }
    if (breakout_token_predict_with_history(
            policy,
            obs,
            policy->prev_actions,
            policy->prev_action_count,
            policy->prev_obs_values,
            policy->prev_obs_count,
            out_action) != TKM_OK) {
        return TKM_ERR;
    }
    breakout_token_obs_history_push(
        policy->prev_obs_values,
        &policy->prev_obs_count,
        current_obs_values,
        policy->selected_dim_count);
    breakout_token_history_push(policy->prev_actions, &policy->prev_action_count, *out_action);
    return TKM_OK;
}
